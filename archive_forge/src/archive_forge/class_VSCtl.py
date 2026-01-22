import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
class VSCtl(object):
    """
    A class to describe an Open vSwitch instance.

    ``remote`` specifies the address of the OVS instance.
    :py:mod:`os_ken.lib.ovs.vsctl.valid_ovsdb_addr` is a convenient function to
    validate this address.
    """

    def _reset(self):
        self.schema_helper = None
        self.ovs = None
        self.txn = None
        self.wait_for_reload = True
        self.dry_run = False

    def __init__(self, remote):
        super(VSCtl, self).__init__()
        self.remote = remote
        self.schema_json = None
        self.schema = None
        self.schema_helper = None
        self.ovs = None
        self.txn = None
        self.wait_for_reload = True
        self.dry_run = False

    def _rpc_get_schema_json(self, database):
        LOG.debug('remote %s', self.remote)
        error, stream_ = stream.Stream.open_block(stream.Stream.open(self.remote))
        if error:
            vsctl_fatal('error %s' % os.strerror(error))
        rpc = jsonrpc.Connection(stream_)
        request = jsonrpc.Message.create_request('get_schema', [database])
        error, reply = rpc.transact_block(request)
        rpc.close()
        if error:
            vsctl_fatal(os.strerror(error))
        elif reply.error:
            vsctl_fatal('error %s' % reply.error)
        return reply.result

    def _init_schema_helper(self):
        if self.schema_json is None:
            self.schema_json = self._rpc_get_schema_json(vswitch_idl.OVSREC_DB_NAME)
            schema_helper = idl.SchemaHelper(None, self.schema_json)
            schema_helper.register_all()
            self.schema = schema_helper.get_idl_schema()
        self.schema_helper = idl.SchemaHelper(None, self.schema_json)

    @staticmethod
    def _idl_block(idl_):
        poller = ovs.poller.Poller()
        idl_.wait(poller)
        poller.block()

    @staticmethod
    def _idl_wait(idl_, seqno):
        while idl_.change_seqno == seqno and (not idl_.run()):
            VSCtl._idl_block(idl_)

    def _run_prerequisites(self, commands):
        schema_helper = self.schema_helper
        schema_helper.register_table(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH)
        if self.wait_for_reload:
            schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, [vswitch_idl.OVSREC_OPEN_VSWITCH_COL_CUR_CFG])
        for command in commands:
            if not command._prerequisite:
                continue
            ctx = VSCtlContext(None, None, None)
            command._prerequisite(ctx, command)
            ctx.done()

    def _do_vsctl(self, idl_, commands):
        self.txn = idl.Transaction(idl_)
        if self.dry_run:
            self.txn.dry_run = True
        self.txn.add_comment('ovs-vsctl')
        ovs_rows = idl_.tables[vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH].rows
        if ovs_rows:
            ovs_ = list(ovs_rows.values())[0]
        else:
            ovs_ = self.txn.insert(idl_.tables[vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH])
        if self.wait_for_reload:
            ovs_.increment(vswitch_idl.OVSREC_OPEN_VSWITCH_COL_NEXT_CFG)
        ctx = VSCtlContext(idl_, self.txn, ovs_)
        for command in commands:
            if not command._run:
                continue
            command._run(ctx, command)
            if ctx.try_again:
                return False
        LOG.debug('result:\n%s', [command.result for command in commands])
        ctx.done()
        status = self.txn.commit_block()
        next_cfg = 0
        if self.wait_for_reload and status == idl.Transaction.SUCCESS:
            next_cfg = self.txn.get_increment_new_value()
        txn_ = self.txn
        self.txn = None
        if status in (idl.Transaction.UNCOMMITTED, idl.Transaction.INCOMPLETE):
            not_reached()
        elif status == idl.Transaction.ABORTED:
            vsctl_fatal('transaction aborted')
        elif status == idl.Transaction.UNCHANGED:
            LOG.debug('unchanged')
        elif status == idl.Transaction.SUCCESS:
            LOG.debug('success')
        elif status == idl.Transaction.TRY_AGAIN:
            return False
        elif status == idl.Transaction.ERROR:
            vsctl_fatal('transaction error: %s' % txn_.get_error())
        elif status == idl.Transaction.NOT_LOCKED:
            vsctl_fatal('database not locked')
        else:
            not_reached()
        if self.wait_for_reload and status != idl.Transaction.UNCHANGED:
            while True:
                idl_.run()
                if ovs_.cur_cfg >= next_cfg:
                    break
                self._idl_block(idl_)
        return True

    def _do_main(self, commands):
        """
        :type commands: list of VSCtlCommand
        """
        self._reset()
        self._init_schema_helper()
        self._run_prerequisites(commands)
        idl_ = idl.Idl(self.remote, self.schema_helper)
        seqno = idl_.change_seqno
        while True:
            self._idl_wait(idl_, seqno)
            seqno = idl_.change_seqno
            if self._do_vsctl(idl_, commands):
                break
            if self.txn:
                self.txn.abort()
                self.txn = None
        idl_.close()

    def _run_command(self, commands):
        """
        :type commands: list of VSCtlCommand
        """
        all_commands = {'init': (None, self._cmd_init), 'show': (self._pre_cmd_show, self._cmd_show), 'add-br': (self._pre_add_br, self._cmd_add_br), 'del-br': (self._pre_get_info, self._cmd_del_br), 'list-br': (self._pre_get_info, self._cmd_list_br), 'br-exists': (self._pre_get_info, self._cmd_br_exists), 'br-to-vlan': (self._pre_get_info, self._cmd_br_to_vlan), 'br-to-parent': (self._pre_get_info, self._cmd_br_to_parent), 'br-set-external-id': (self._pre_cmd_br_set_external_id, self._cmd_br_set_external_id), 'br-get-external-id': (self._pre_cmd_br_get_external_id, self._cmd_br_get_external_id), 'list-ports': (self._pre_get_info, self._cmd_list_ports), 'add-port': (self._pre_cmd_add_port, self._cmd_add_port), 'add-bond': (self._pre_cmd_add_bond, self._cmd_add_bond), 'del-port': (self._pre_get_info, self._cmd_del_port), 'port-to-br': (self._pre_get_info, self._cmd_port_to_br), 'list-ifaces': (self._pre_get_info, self._cmd_list_ifaces), 'iface-to-br': (self._pre_get_info, self._cmd_iface_to_br), 'get-controller': (self._pre_controller, self._cmd_get_controller), 'del-controller': (self._pre_controller, self._cmd_del_controller), 'set-controller': (self._pre_controller, self._cmd_set_controller), 'get-fail-mode': (self._pre_fail_mode, self._cmd_get_fail_mode), 'del-fail-mode': (self._pre_fail_mode, self._cmd_del_fail_mode), 'set-fail-mode': (self._pre_fail_mode, self._cmd_set_fail_mode), 'list': (self._pre_cmd_list, self._cmd_list), 'find': (self._pre_cmd_find, self._cmd_find), 'get': (self._pre_cmd_get, self._cmd_get), 'set': (self._pre_cmd_set, self._cmd_set), 'add': (self._pre_cmd_add, self._cmd_add), 'remove': (self._pre_cmd_remove, self._cmd_remove), 'clear': (self._pre_cmd_clear, self._cmd_clear), 'set-qos': (self._pre_cmd_set_qos, self._cmd_set_qos), 'set-queue': (self._pre_cmd_set_queue, self._cmd_set_queue), 'del-qos': (self._pre_get_info, self._cmd_del_qos), 'list-ifaces-verbose': (self._pre_cmd_list_ifaces_verbose, self._cmd_list_ifaces_verbose)}
        for command in commands:
            funcs = all_commands[command.command]
            command._prerequisite, command._run = funcs
        self._do_main(commands)

    def run_command(self, commands, timeout_sec=None, exception=None):
        """
        Executes the given commands and sends OVSDB messages.

        ``commands`` must be a list of
        :py:mod:`os_ken.lib.ovs.vsctl.VSCtlCommand`.

        If ``timeout_sec`` is specified, raises exception after the given
        timeout [sec]. Additionally, if ``exception`` is specified, this
        function will wraps exception using the given exception class.

        Retruns ``None`` but fills ``result`` attribute for each command
        instance.
        """
        if timeout_sec is None:
            self._run_command(commands)
        else:
            with hub.Timeout(timeout_sec, exception):
                self._run_command(commands)

    def _cmd_init(self, _ctx, _command):
        pass
    _CMD_SHOW_TABLES = [_CmdShowTable(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, None, [vswitch_idl.OVSREC_OPEN_VSWITCH_COL_MANAGER_OPTIONS, vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES, vswitch_idl.OVSREC_OPEN_VSWITCH_COL_OVS_VERSION], False), _CmdShowTable(vswitch_idl.OVSREC_TABLE_BRIDGE, vswitch_idl.OVSREC_BRIDGE_COL_NAME, [vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE, vswitch_idl.OVSREC_BRIDGE_COL_PORTS], False), _CmdShowTable(vswitch_idl.OVSREC_TABLE_PORT, vswitch_idl.OVSREC_PORT_COL_NAME, [vswitch_idl.OVSREC_PORT_COL_TAG, vswitch_idl.OVSREC_PORT_COL_TRUNKS, vswitch_idl.OVSREC_PORT_COL_INTERFACES], False), _CmdShowTable(vswitch_idl.OVSREC_TABLE_INTERFACE, vswitch_idl.OVSREC_INTERFACE_COL_NAME, [vswitch_idl.OVSREC_INTERFACE_COL_TYPE, vswitch_idl.OVSREC_INTERFACE_COL_OPTIONS], False), _CmdShowTable(vswitch_idl.OVSREC_TABLE_CONTROLLER, vswitch_idl.OVSREC_CONTROLLER_COL_TARGET, [vswitch_idl.OVSREC_CONTROLLER_COL_IS_CONNECTED], False), _CmdShowTable(vswitch_idl.OVSREC_TABLE_MANAGER, vswitch_idl.OVSREC_MANAGER_COL_TARGET, [vswitch_idl.OVSREC_MANAGER_COL_IS_CONNECTED], False)]

    def _pre_cmd_show(self, _ctx, _command):
        schema_helper = self.schema_helper
        for show in self._CMD_SHOW_TABLES:
            schema_helper.register_table(show.table)
            if show.name_column:
                schema_helper.register_columns(show.table, [show.name_column])
            schema_helper.register_columns(show.table, show.columns)

    @staticmethod
    def _cmd_show_find_table_by_row(row):
        for show in VSCtl._CMD_SHOW_TABLES:
            if show.table == row._table.name:
                return show
        return None

    @staticmethod
    def _cmd_show_find_table_by_name(name):
        for show in VSCtl._CMD_SHOW_TABLES:
            if show.table == name:
                return show
        return None

    @staticmethod
    def _cmd_show_row(ctx, row, level):
        _INDENT_SIZE = 4
        show = VSCtl._cmd_show_find_table_by_row(row)
        output = ''
        output += ' ' * level * _INDENT_SIZE
        if show and show.name_column:
            output += '%s ' % show.table
            datum = getattr(row, show.name_column)
            output += datum
        else:
            output += str(row.uuid)
        output += '\n'
        if not show or show.recurse:
            return
        show.recurse = True
        for column in show.columns:
            datum = row._data[column]
            key = datum.type.key
            if key.type == ovs.db.types.UuidType and key.ref_table_name:
                ref_show = VSCtl._cmd_show_find_table_by_name(key.ref_table_name)
                if ref_show:
                    for atom in datum.values:
                        ref_row = ctx.idl.tables[ref_show.table].rows.get(atom.value)
                        if ref_row:
                            VSCtl._cmd_show_row(ctx, ref_row, level + 1)
                    continue
            if not datum.is_default():
                output += ' ' * (level + 1) * _INDENT_SIZE
                output += '%s: %s\n' % (column, datum)
        show.recurse = False
        return output

    def _cmd_show(self, ctx, command):
        for row in ctx.idl.tables[self._CMD_SHOW_TABLES[0].table].rows.values():
            output = self._cmd_show_row(ctx, row, 0)
            command.result = output

    def _pre_get_info(self, _ctx, _command):
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, [vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_BRIDGE, [vswitch_idl.OVSREC_BRIDGE_COL_NAME, vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE, vswitch_idl.OVSREC_BRIDGE_COL_PORTS])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, [vswitch_idl.OVSREC_PORT_COL_NAME, vswitch_idl.OVSREC_PORT_COL_FAKE_BRIDGE, vswitch_idl.OVSREC_PORT_COL_TAG, vswitch_idl.OVSREC_PORT_COL_INTERFACES, vswitch_idl.OVSREC_PORT_COL_QOS])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_INTERFACE, [vswitch_idl.OVSREC_INTERFACE_COL_NAME])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QOS, [vswitch_idl.OVSREC_QOS_COL_QUEUES])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QUEUE, [])

    def _cmd_list_br(self, ctx, command):
        ctx.populate_cache()
        command.result = sorted(ctx.bridges.keys())

    def _pre_add_br(self, ctx, command):
        self._pre_get_info(ctx, command)
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_INTERFACE, [vswitch_idl.OVSREC_INTERFACE_COL_TYPE])

    def _cmd_add_br(self, ctx, command):
        br_name = command.args[0]
        parent_name = None
        vlan = 0
        if len(command.args) == 1:
            pass
        elif len(command.args) == 3:
            parent_name = command.args[1]
            vlan = int(command.args[2])
            if vlan < 0 or vlan > 4095:
                vsctl_fatal('vlan must be between 0 and 4095 %d' % vlan)
        else:
            vsctl_fatal('this command takes exactly 1 or 3 argument')
        ctx.add_bridge(br_name, parent_name, vlan)

    def _del_br(self, ctx, br_name, must_exist=False):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, must_exist)
        if br:
            ctx.del_bridge(br)

    def _cmd_del_br(self, ctx, command):
        br_name = command.args[0]
        self._del_br(ctx, br_name)

    def _br_exists(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, must_exist=False)
        return br is not None

    def _cmd_br_exists(self, ctx, command):
        br_name = command.args[0]
        command.result = self._br_exists(ctx, br_name)

    def _br_to_vlan(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, must_exist=True)
        vlan = br.vlan
        if isinstance(vlan, list):
            if len(vlan) == 0:
                vlan = 0
            else:
                vlan = vlan[0]
        return vlan

    def _cmd_br_to_vlan(self, ctx, command):
        br_name = command.args[0]
        command.result = self._br_to_vlan(ctx, br_name)

    def _br_to_parent(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, must_exist=True)
        return br if br.parent is None else br.parent

    def _cmd_br_to_parent(self, ctx, command):
        br_name = command.args[0]
        command.result = self._br_to_parent(ctx, br_name)

    def _pre_cmd_br_set_external_id(self, ctx, _command):
        table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
        columns = [vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS]
        self._pre_mod_columns(ctx, table_name, columns)

    def _br_add_external_id(self, ctx, br_name, key, value):
        table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
        column = vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, br_name)
        value_json = ['map', [[key, value]]]
        ctx.add_column(ovsrec_row, column, value_json)
        ctx.invalidate_cache()

    def _br_clear_external_id(self, ctx, br_name, key):
        table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
        column = vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, br_name)
        values = getattr(ovsrec_row, column, {})
        values.pop(key, None)
        setattr(ovsrec_row, column, values)
        ctx.invalidate_cache()

    def _cmd_br_set_external_id(self, ctx, command):
        br_name = command.args[0]
        key = command.args[1]
        if len(command.args) > 2:
            self._br_add_external_id(ctx, br_name, key, command.args[2])
        else:
            self._br_clear_external_id(ctx, br_name, key)

    def _pre_cmd_br_get_external_id(self, ctx, _command):
        table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
        columns = [vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS]
        self._pre_get_columns(ctx, table_name, columns)

    def _br_get_external_id_value(self, ctx, br_name, key):
        external_id = self._br_get_external_id_list(ctx, br_name)
        return external_id.get(key, None)

    def _br_get_external_id_list(self, ctx, br_name):
        table_name = vswitch_idl.OVSREC_TABLE_BRIDGE
        column = vswitch_idl.OVSREC_BRIDGE_COL_EXTERNAL_IDS
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, br_name)
        return ctx.get_column(ovsrec_row, column)

    def _cmd_br_get_external_id(self, ctx, command):
        br_name = command.args[0]
        if len(command.args) > 1:
            command.result = self._br_get_external_id_value(ctx, br_name, command.args[1])
        else:
            command.result = self._br_get_external_id_list(ctx, br_name)

    def _list_ports(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        if br.br_cfg:
            br.br_cfg.verify(vswitch_idl.OVSREC_BRIDGE_COL_PORTS)
        else:
            br.parent.br_cfg.verify(vswitch_idl.OVSREC_BRIDGE_COL_PORTS)
        return [port.port_cfg.name for port in br.ports if port.port_cfg.name != br.name]

    def _cmd_list_ports(self, ctx, command):
        br_name = command.args[0]
        port_names = self._list_ports(ctx, br_name)
        command.result = sorted(port_names)

    def _pre_add_port(self, _ctx, columns):
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, [vswitch_idl.OVSREC_PORT_COL_NAME, vswitch_idl.OVSREC_PORT_COL_BOND_FAKE_IFACE])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_PORT, columns)

    def _pre_cmd_add_port(self, ctx, command):
        self._pre_get_info(ctx, command)
        columns = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting)[0] for setting in command.args[2:]]
        self._pre_add_port(ctx, columns)

    def _pre_cmd_add_bond(self, ctx, command):
        self._pre_get_info(ctx, command)
        if len(command.args) < 3:
            vsctl_fatal('this command requires at least 3 arguments')
        columns = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting)[0] for setting in command.args[3:]]
        self._pre_add_port(ctx, columns)

    def _cmd_add_port(self, ctx, command):
        may_exist = command.has_option('--may_exist') or command.has_option('--may-exist')
        br_name = command.args[0]
        port_name = command.args[1]
        iface_names = [command.args[1]]
        settings = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting) for setting in command.args[2:]]
        ctx.add_port(br_name, port_name, may_exist, False, iface_names, settings)

    def _cmd_add_bond(self, ctx, command):
        may_exist = command.has_option('--may_exist') or command.has_option('--may-exist')
        fake_iface = command.has_option('--fake-iface')
        br_name = command.args[0]
        port_name = command.args[1]
        iface_names = list(command.args[2])
        settings = [ctx.parse_column_key_value(self.schema.tables[vswitch_idl.OVSREC_TABLE_PORT], setting) for setting in command.args[3:]]
        ctx.add_port(br_name, port_name, may_exist, fake_iface, iface_names, settings)

    def _del_port(self, ctx, br_name=None, target=None, must_exist=False, with_iface=False):
        assert target is not None
        ctx.populate_cache()
        if not with_iface:
            vsctl_port = ctx.find_port(target, must_exist)
        else:
            vsctl_port = ctx.find_port(target, False)
            if not vsctl_port:
                vsctl_iface = ctx.find_iface(target, False)
                if vsctl_iface:
                    vsctl_port = vsctl_iface.port()
                if must_exist and (not vsctl_port):
                    vsctl_fatal('no port or interface named %s' % target)
        if not vsctl_port:
            return
        if not br_name:
            vsctl_bridge = ctx.find_bridge(br_name, True)
            if vsctl_port.bridge() != vsctl_bridge:
                if vsctl_port.bridge().parent == vsctl_bridge:
                    vsctl_fatal('bridge %s does not have a port %s (although its parent bridge %s does)' % (br_name, target, vsctl_bridge.parent.name))
                else:
                    vsctl_fatal('bridge %s does not have a port %s' % (br_name, target))
        ctx.del_port(vsctl_port)

    def _cmd_del_port(self, ctx, command):
        must_exist = command.has_option('--must-exist')
        with_iface = command.has_option('--with-iface')
        target = command.args[-1]
        br_name = command.args[0] if len(command.args) == 2 else None
        self._del_port(ctx, br_name, target, must_exist, with_iface)

    def _port_to_br(self, ctx, port_name):
        ctx.populate_cache()
        port = ctx.find_port(port_name, True)
        bridge = port.bridge()
        if bridge is None:
            vsctl_fatal('Bridge associated to port "%s" does not exist' % port_name)
        return bridge.name

    def _cmd_port_to_br(self, ctx, command):
        iface_name = command.args[0]
        command.result = self._iface_to_br(ctx, iface_name)

    def _list_ifaces(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        ctx.verify_ports()
        iface_names = set()
        for vsctl_port in br.ports:
            for vsctl_iface in vsctl_port.ifaces:
                iface_name = vsctl_iface.iface_cfg.name
                if iface_name != br_name:
                    iface_names.add(iface_name)
        return iface_names

    def _cmd_list_ifaces(self, ctx, command):
        br_name = command.args[0]
        iface_names = self._list_ifaces(ctx, br_name)
        command.result = sorted(iface_names)

    def _iface_to_br(self, ctx, iface_name):
        ctx.populate_cache()
        iface = ctx.find_iface(iface_name, True)
        port = iface.port()
        if port is None:
            vsctl_fatal('Port associated to iface "%s" does not exist' % iface_name)
        bridge = port.bridge()
        if bridge is None:
            vsctl_fatal('Bridge associated to iface "%s" does not exist' % iface_name)
        return bridge.name

    def _cmd_iface_to_br(self, ctx, command):
        iface_name = command.args[0]
        command.result = self._iface_to_br(ctx, iface_name)

    def _pre_cmd_list_ifaces_verbose(self, ctx, command):
        self._pre_get_info(ctx, command)
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_BRIDGE, [vswitch_idl.OVSREC_BRIDGE_COL_DATAPATH_ID])
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_INTERFACE, [vswitch_idl.OVSREC_INTERFACE_COL_TYPE, vswitch_idl.OVSREC_INTERFACE_COL_NAME, vswitch_idl.OVSREC_INTERFACE_COL_EXTERNAL_IDS, vswitch_idl.OVSREC_INTERFACE_COL_OPTIONS, vswitch_idl.OVSREC_INTERFACE_COL_OFPORT])

    @staticmethod
    def _iface_to_dict(iface_cfg):
        _ATTRIBUTE = ['name', 'ofport', 'type', 'external_ids', 'options']
        attr = dict(((key, getattr(iface_cfg, key)) for key in _ATTRIBUTE))
        if attr['ofport']:
            attr['ofport'] = attr['ofport'][0]
        return attr

    def _list_ifaces_verbose(self, ctx, datapath_id, port_name):
        ctx.populate_cache()
        br = ctx.find_bridge_by_id(datapath_id, True)
        ctx.verify_ports()
        iface_cfgs = []
        if port_name is None:
            for vsctl_port in br.ports:
                iface_cfgs.extend((self._iface_to_dict(vsctl_iface.iface_cfg) for vsctl_iface in vsctl_port.ifaces))
        else:
            for vsctl_port in br.ports:
                iface_cfgs.extend((self._iface_to_dict(vsctl_iface.iface_cfg) for vsctl_iface in vsctl_port.ifaces if vsctl_iface.iface_cfg.name == port_name))
        return iface_cfgs

    def _cmd_list_ifaces_verbose(self, ctx, command):
        datapath_id = command.args[0]
        port_name = None
        if len(command.args) >= 2:
            port_name = command.args[1]
        LOG.debug('command.args %s', command.args)
        iface_cfgs = self._list_ifaces_verbose(ctx, datapath_id, port_name)
        command.result = sorted(iface_cfgs)

    def _verify_controllers(self, ovsrec_bridge):
        ovsrec_bridge.verify(vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER)
        for controller in ovsrec_bridge.controller:
            controller.verify(vswitch_idl.OVSREC_CONTROLLER_COL_TARGET)

    def _pre_controller(self, ctx, command):
        self._pre_get_info(ctx, command)
        self.schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_CONTROLLER, [vswitch_idl.OVSREC_CONTROLLER_COL_TARGET])

    def _get_controller(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        self._verify_controllers(br.br_cfg)
        return set((controller.target for controller in br.br_cfg.controller))

    def _cmd_get_controller(self, ctx, command):
        br_name = command.args[0]
        controller_names = self._get_controller(ctx, br_name)
        command.result = sorted(controller_names)

    def _delete_controllers(self, ovsrec_controllers):
        for controller in ovsrec_controllers:
            controller.delete()

    def _del_controller(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_real_bridge(br_name, True)
        ovsrec_bridge = br.br_cfg
        self._verify_controllers(ovsrec_bridge)
        if ovsrec_bridge.controller:
            self._delete_controllers(ovsrec_bridge.controller)
            ovsrec_bridge.controller = []

    def _cmd_del_controller(self, ctx, command):
        br_name = command.args[0]
        self._del_controller(ctx, br_name)

    def _insert_controllers(self, controller_names):
        ovsrec_controllers = []
        for name in controller_names:
            ovsrec_controller = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_CONTROLLER])
            ovsrec_controller.target = name
            ovsrec_controllers.append(ovsrec_controller)
        return ovsrec_controllers

    def _insert_qos(self):
        ovsrec_qos = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_QOS])
        return ovsrec_qos

    def _set_controller(self, ctx, br_name, controller_names):
        ctx.populate_cache()
        ovsrec_bridge = ctx.find_real_bridge(br_name, True).br_cfg
        self._verify_controllers(ovsrec_bridge)
        self._delete_controllers(ovsrec_bridge.controller)
        controllers = self._insert_controllers(controller_names)
        ovsrec_bridge.controller = controllers

    def _cmd_set_controller(self, ctx, command):
        br_name = command.args[0]
        controller_names = command.args[1:]
        self._set_controller(ctx, br_name, controller_names)

    def _pre_fail_mode(self, ctx, command):
        self._pre_get_info(ctx, command)
        self.schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_BRIDGE, [vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE])

    def _get_fail_mode(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        return getattr(br.br_cfg, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE)[0]

    def _cmd_get_fail_mode(self, ctx, command):
        br_name = command.args[0]
        command.result = self._get_fail_mode(ctx, br_name)

    def _del_fail_mode(self, ctx, br_name):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        setattr(br.br_cfg, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE, [])
        ctx.invalidate_cache()

    def _cmd_del_fail_mode(self, ctx, command):
        br_name = command.args[0]
        self._del_fail_mode(ctx, br_name)

    def _set_fail_mode(self, ctx, br_name, mode):
        ctx.populate_cache()
        br = ctx.find_bridge(br_name, True)
        setattr(br.br_cfg, vswitch_idl.OVSREC_BRIDGE_COL_FAIL_MODE, mode)
        ctx.invalidate_cache()

    def _cmd_set_fail_mode(self, ctx, command):
        br_name = command.args[0]
        mode = command.args[1]
        if mode not in ('standalone', 'secure'):
            vsctl_fatal('fail-mode must be "standalone" or "secure"')
        self._set_fail_mode(ctx, br_name, mode)

    def _del_qos(self, ctx, port_name):
        assert port_name is not None
        ctx.populate_cache()
        vsctl_port = ctx.find_port(port_name, True)
        vsctl_qos = vsctl_port.qos
        ctx.del_qos(vsctl_qos)

    def _cmd_del_qos(self, ctx, command):
        port_name = command.args[0]
        self._del_qos(ctx, port_name)

    def _set_qos(self, ctx, port_name, type, max_rate):
        ctx.populate_cache()
        vsctl_port = ctx.find_port(port_name, True)
        ovsrec_qos = ctx.set_qos(vsctl_port, type, max_rate)
        return ovsrec_qos

    def _cmd_set_qos(self, ctx, command):
        port_name = command.args[0]
        type = command.args[1]
        max_rate = command.args[2]
        result = self._set_qos(ctx, port_name, type, max_rate)
        command.result = [result]

    def _pre_cmd_set_qos(self, ctx, command):
        self._pre_get_info(ctx, command)
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QOS, [vswitch_idl.OVSREC_QOS_COL_EXTERNAL_IDS, vswitch_idl.OVSREC_QOS_COL_OTHER_CONFIG, vswitch_idl.OVSREC_QOS_COL_QUEUES, vswitch_idl.OVSREC_QOS_COL_TYPE])

    def _cmd_set_queue(self, ctx, command):
        ctx.populate_cache()
        port_name = command.args[0]
        queues = command.args[1]
        vsctl_port = ctx.find_port(port_name, True)
        vsctl_qos = vsctl_port.qos
        queue_id = 0
        results = []
        for queue in queues:
            max_rate = queue.get('max-rate', None)
            min_rate = queue.get('min-rate', None)
            ovsrec_queue = ctx.set_queue(vsctl_qos, max_rate, min_rate, queue_id)
            results.append(ovsrec_queue)
            queue_id += 1
        command.result = results

    def _pre_cmd_set_queue(self, ctx, command):
        self._pre_get_info(ctx, command)
        schema_helper = self.schema_helper
        schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QUEUE, [vswitch_idl.OVSREC_QUEUE_COL_DSCP, vswitch_idl.OVSREC_QUEUE_COL_EXTERNAL_IDS, vswitch_idl.OVSREC_QUEUE_COL_OTHER_CONFIG])
    _TABLES = [_VSCtlTable(vswitch_idl.OVSREC_TABLE_BRIDGE, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_BRIDGE, vswitch_idl.OVSREC_BRIDGE_COL_NAME, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_CONTROLLER, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_BRIDGE, vswitch_idl.OVSREC_BRIDGE_COL_NAME, vswitch_idl.OVSREC_BRIDGE_COL_CONTROLLER)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_INTERFACE, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_INTERFACE, vswitch_idl.OVSREC_INTERFACE_COL_NAME, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_MIRROR, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_MIRROR, vswitch_idl.OVSREC_MIRROR_COL_NAME, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_MANAGER, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_MANAGER, vswitch_idl.OVSREC_MANAGER_COL_TARGET, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_NETFLOW, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_BRIDGE, vswitch_idl.OVSREC_BRIDGE_COL_NAME, vswitch_idl.OVSREC_BRIDGE_COL_NETFLOW)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, None, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_PORT, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_PORT, vswitch_idl.OVSREC_PORT_COL_NAME, None)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_QOS, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_PORT, vswitch_idl.OVSREC_PORT_COL_NAME, vswitch_idl.OVSREC_PORT_COL_QOS)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_QUEUE, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_QOS, None, vswitch_idl.OVSREC_QOS_COL_QUEUES)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_SSL, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH, None, vswitch_idl.OVSREC_OPEN_VSWITCH_COL_SSL)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_SFLOW, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_BRIDGE, vswitch_idl.OVSREC_BRIDGE_COL_NAME, vswitch_idl.OVSREC_BRIDGE_COL_SFLOW)]), _VSCtlTable(vswitch_idl.OVSREC_TABLE_FLOW_TABLE, [_VSCtlRowID(vswitch_idl.OVSREC_TABLE_FLOW_TABLE, vswitch_idl.OVSREC_FLOW_TABLE_COL_NAME, None)])]

    @staticmethod
    def _score_partial_match(name, s):
        _MAX_SCORE = 4294967295
        assert len(name) < _MAX_SCORE
        s = s[:_MAX_SCORE - 1]
        if name == s:
            return _MAX_SCORE
        name = name.lower().replace('-', '_')
        s = s.lower().replace('-', '_')
        if s.startswith(name):
            return _MAX_SCORE - 1
        if name.startswith(s):
            return len(s)
        return 0

    @staticmethod
    def _get_table(table_name):
        best_match = None
        best_score = 0
        for table in VSCtl._TABLES:
            score = VSCtl._score_partial_match(table.table_name, table_name)
            if score > best_score:
                best_match = table
                best_score = score
            elif score == best_score:
                best_match = None
        if best_match:
            return best_match
        elif best_score:
            vsctl_fatal('multiple table names match "%s"' % table_name)
        else:
            vsctl_fatal('unknown table "%s"' % table_name)

    def _pre_get_table(self, _ctx, table_name):
        vsctl_table = self._get_table(table_name)
        schema_helper = self.schema_helper
        schema_helper.register_table(vsctl_table.table_name)
        for row_id in vsctl_table.row_ids:
            if row_id.table:
                schema_helper.register_table(row_id.table)
            if row_id.name_column:
                schema_helper.register_columns(row_id.table, [row_id.name_column])
            if row_id.uuid_column:
                schema_helper.register_columns(row_id.table, [row_id.uuid_column])
        return vsctl_table

    def _get_column(self, table_name, column_name):
        best_match = None
        best_score = 0
        columns = self.schema.tables[table_name].columns.keys()
        for column in columns:
            score = VSCtl._score_partial_match(column, column_name)
            if score > best_score:
                best_match = column
                best_score = score
            elif score == best_score:
                best_match = None
        if best_match:
            return str(best_match)
        elif best_score:
            vsctl_fatal('%s contains more than one column whose name matches "%s"' % (table_name, column_name))
        else:
            vsctl_fatal('%s does not contain a column whose name matches "%s"' % (table_name, column_name))

    def _pre_get_column(self, _ctx, table_name, column):
        column_name = self._get_column(table_name, column)
        self.schema_helper.register_columns(table_name, [column_name])

    def _pre_get_columns(self, ctx, table_name, columns):
        self._pre_get_table(ctx, table_name)
        for column in columns:
            self._pre_get_column(ctx, table_name, column)

    def _pre_cmd_list(self, ctx, command):
        table_name = command.args[0]
        self._pre_get_table(ctx, table_name)

    def _list(self, ctx, table_name, record_id=None):
        result = []
        for ovsrec_row in ctx.idl.tables[table_name].rows.values():
            if record_id is not None and ovsrec_row.name != record_id:
                continue
            result.append(ovsrec_row)
        return result

    def _cmd_list(self, ctx, command):
        table_name = command.args[0]
        record_id = None
        if len(command.args) > 1:
            record_id = command.args[1]
        command.result = self._list(ctx, table_name, record_id)

    def _pre_cmd_find(self, ctx, command):
        table_name = command.args[0]
        table_schema = self.schema.tables[table_name]
        columns = [ctx.parse_column_key_value(table_schema, column_key_value)[0] for column_key_value in command.args[1:]]
        self._pre_get_columns(ctx, table_name, columns)

    def _check_value(self, ovsrec_row, column_value):
        """
        :type column_value: tuple of column and value_json
        """
        column, value_json = column_value
        column_schema = ovsrec_row._table.columns[column]
        value = ovs.db.data.Datum.from_json(column_schema.type, value_json).to_python(ovs.db.idl._uuid_to_row)
        datum = getattr(ovsrec_row, column)
        if column_schema.type.is_map():
            for k, v in value.items():
                if k in datum and datum[k] == v:
                    return True
        elif datum == value:
            return True
        return False

    def _find(self, ctx, table_name, column_values):
        """
        :type column_values: list of (column, value_json)
        """
        result = []
        for ovsrec_row in ctx.idl.tables[table_name].rows.values():
            LOG.debug('ovsrec_row %s', ovsrec_row_to_string(ovsrec_row))
            if all((self._check_value(ovsrec_row, column_value) for column_value in column_values)):
                result.append(ovsrec_row)
        return result

    def _cmd_find(self, ctx, command):
        table_name = command.args[0]
        table_schema = self.schema.tables[table_name]
        column_values = [ctx.parse_column_key_value(table_schema, column_key_value) for column_key_value in command.args[1:]]
        command.result = self._find(ctx, table_name, column_values)

    def _pre_cmd_get(self, ctx, command):
        table_name = command.args[0]
        columns = [ctx.parse_column_key(column_key)[0] for column_key in command.args[2:]]
        self._pre_get_columns(ctx, table_name, columns)

    def _get(self, ctx, table_name, record_id, column_keys, id_=None, if_exists=False):
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, record_id)
        result = []
        for column, key in column_keys:
            result.append(ctx.get_column(ovsrec_row, column, key, if_exists))
        return result

    def _cmd_get(self, ctx, command):
        id_ = None
        if_exists = command.has_option('--if-exists')
        table_name = command.args[0]
        record_id = command.args[1]
        column_keys = [ctx.parse_column_key(column_key) for column_key in command.args[2:]]
        command.result = self._get(ctx, table_name, record_id, column_keys, id_, if_exists)

    def _check_mutable(self, table_name, column):
        column_schema = self.schema.tables[table_name].columns[column]
        if not column_schema.mutable:
            vsctl_fatal('cannot modify read-only column %s in table %s' % (column, table_name))

    def _pre_mod_columns(self, ctx, table_name, columns):
        self._pre_get_table(ctx, table_name)
        for column in columns:
            self._pre_get_column(ctx, table_name, column)
            self._check_mutable(table_name, column)

    def _pre_cmd_set(self, ctx, command):
        table_name = command.args[0]
        table_schema = self.schema.tables[table_name]
        columns = [ctx.parse_column_key_value(table_schema, column_key_value)[0] for column_key_value in command.args[2:]]
        self._pre_mod_columns(ctx, table_name, columns)

    def _set(self, ctx, table_name, record_id, column_values):
        """
        :type column_values: list of (column, value_json)
        """
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, record_id)
        for column, value in column_values:
            ctx.set_column(ovsrec_row, column, value)
        ctx.invalidate_cache()

    def _cmd_set(self, ctx, command):
        table_name = command.args[0]
        record_id = command.args[1]
        table_schema = self.schema.tables[table_name]
        column_values = [ctx.parse_column_key_value(table_schema, column_key_value) for column_key_value in command.args[2:]]
        self._set(ctx, table_name, record_id, column_values)

    def _pre_cmd_add(self, ctx, command):
        table_name = command.args[0]
        columns = [command.args[2]]
        self._pre_mod_columns(ctx, table_name, columns)

    def _add(self, ctx, table_name, record_id, column_values):
        """
        :type column_values: list of (column, value_json)
        """
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, record_id)
        for column, value in column_values:
            ctx.add_column(ovsrec_row, column, value)
        ctx.invalidate_cache()

    def _cmd_add(self, ctx, command):
        table_name = command.args[0]
        record_id = command.args[1]
        column = command.args[2]
        column_key_value_strings = []
        for value in command.args[3:]:
            if '=' in value:
                column_key_value_strings.append('%s:%s' % (column, value))
            else:
                column_key_value_strings.append('%s=%s' % (column, value))
        table_schema = self.schema.tables[table_name]
        column_values = [ctx.parse_column_key_value(table_schema, column_key_value_string) for column_key_value_string in column_key_value_strings]
        self._add(ctx, table_name, record_id, column_values)

    def _pre_cmd_remove(self, ctx, command):
        table_name = command.args[0]
        columns = [command.args[2]]
        self._pre_mod_columns(ctx, table_name, columns)

    def _remove(self, ctx, table_name, record_id, column_values):
        """
        :type column_values: list of (column, value_json)
        """
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, record_id)
        for column, value in column_values:
            ctx.remove_column(ovsrec_row, column, value)
        ctx.invalidate_cache()

    def _cmd_remove(self, ctx, command):
        table_name = command.args[0]
        record_id = command.args[1]
        column = command.args[2]
        column_key_value_strings = []
        for value in command.args[3:]:
            if '=' in value:
                column_key_value_strings.append('%s:%s' % (column, value))
            else:
                column_key_value_strings.append('%s=%s' % (column, value))
        table_schema = self.schema.tables[table_name]
        column_values = [ctx.parse_column_key_value(table_schema, column_key_value_string) for column_key_value_string in column_key_value_strings]
        self._remove(ctx, table_name, record_id, column_values)

    def _pre_cmd_clear(self, ctx, command):
        table_name = command.args[0]
        column = command.args[2]
        self._pre_mod_columns(ctx, table_name, [column])

    def _clear(self, ctx, table_name, record_id, column):
        vsctl_table = self._get_table(table_name)
        ovsrec_row = ctx.must_get_row(vsctl_table, record_id)
        column_schema = ctx.idl.tables[table_name].columns[column]
        if column_schema.type.n_min > 0:
            vsctl_fatal('"clear" operation cannot be applied to column %s of table %s, which is not allowed to be empty' % (column, table_name))
        default_datum = ovs.db.data.Datum.default(column_schema.type)
        setattr(ovsrec_row, column, default_datum.to_python(ovs.db.idl._uuid_to_row))
        ctx.invalidate_cache()

    def _cmd_clear(self, ctx, command):
        table_name = command.args[0]
        record_id = command.args[1]
        column = command.args[2]
        self._clear(ctx, table_name, record_id, column)