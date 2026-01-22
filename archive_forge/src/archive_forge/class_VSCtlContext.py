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
class VSCtlContext(object):

    def _invalidate_cache(self):
        self.cache_valid = False
        self.bridges.clear()
        self.ports.clear()
        self.ifaces.clear()

    def __init__(self, idl_, txn, ovsrec_open_vswitch):
        super(VSCtlContext, self).__init__()
        self.idl = idl_
        self.txn = txn
        self.ovs = ovsrec_open_vswitch
        self.symtab = None
        self.verified_ports = False
        self.cache_valid = False
        self.bridges = {}
        self.ports = {}
        self.ifaces = {}
        self.try_again = False

    def done(self):
        self._invalidate_cache()

    def verify_bridges(self):
        self.ovs.verify(vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES)

    def verify_ports(self):
        if self.verified_ports:
            return
        self.verify_bridges()
        for ovsrec_bridge in self.idl.tables[vswitch_idl.OVSREC_TABLE_BRIDGE].rows.values():
            ovsrec_bridge.verify(vswitch_idl.OVSREC_BRIDGE_COL_PORTS)
        for ovsrec_port in self.idl.tables[vswitch_idl.OVSREC_TABLE_PORT].rows.values():
            ovsrec_port.verify(vswitch_idl.OVSREC_PORT_COL_INTERFACES)
        self.verified_ports = True

    def add_bridge_to_cache(self, ovsrec_bridge, name, parent, vlan):
        vsctl_bridge = VSCtlBridge(ovsrec_bridge, name, parent, vlan)
        if parent:
            parent.children.add(vsctl_bridge)
        self.bridges[name] = vsctl_bridge
        return vsctl_bridge

    def del_cached_bridge(self, vsctl_bridge):
        assert not vsctl_bridge.ports
        assert not vsctl_bridge.children
        parent = vsctl_bridge.parent
        if parent:
            parent.children.remove(vsctl_bridge)
            vsctl_bridge.parent = None
        ovsrec_bridge = vsctl_bridge.br_cfg
        if ovsrec_bridge:
            ovsrec_bridge.delete()
            self.ovs_delete_bridge(ovsrec_bridge)
        del self.bridges[vsctl_bridge.name]

    def del_cached_qos(self, vsctl_qos):
        vsctl_qos.port().qos = None
        vsctl_qos.port = None
        vsctl_qos.queues = None

    def add_port_to_cache(self, vsctl_bridge_parent, ovsrec_port):
        tag = getattr(ovsrec_port, vswitch_idl.OVSREC_PORT_COL_TAG, None)
        if isinstance(tag, list):
            if len(tag) == 0:
                tag = 0
            else:
                tag = tag[0]
        if tag is not None and 0 <= tag < 4096:
            vlan_bridge = vsctl_bridge_parent.find_vlan_bridge(tag)
            if vlan_bridge:
                vsctl_bridge_parent = vlan_bridge
        vsctl_port = VSCtlPort(vsctl_bridge_parent, ovsrec_port)
        vsctl_bridge_parent.ports.add(vsctl_port)
        self.ports[ovsrec_port.name] = vsctl_port
        return vsctl_port

    def del_cached_port(self, vsctl_port):
        assert not vsctl_port.ifaces
        vsctl_port.bridge().ports.remove(vsctl_port)
        vsctl_port.bridge = None
        port = self.ports.pop(vsctl_port.port_cfg.name)
        assert port == vsctl_port
        vsctl_port.port_cfg.delete()

    def add_iface_to_cache(self, vsctl_port_parent, ovsrec_iface):
        vsctl_iface = VSCtlIface(vsctl_port_parent, ovsrec_iface)
        vsctl_port_parent.ifaces.add(vsctl_iface)
        self.ifaces[ovsrec_iface.name] = vsctl_iface

    def add_qos_to_cache(self, vsctl_port_parent, ovsrec_qos):
        vsctl_qos = VSCtlQoS(vsctl_port_parent, ovsrec_qos)
        vsctl_port_parent.qos = vsctl_qos
        return vsctl_qos

    def add_queue_to_cache(self, vsctl_qos_parent, ovsrec_queue):
        vsctl_queue = VSCtlQueue(vsctl_qos_parent, ovsrec_queue)
        vsctl_qos_parent.queues.add(vsctl_queue)

    def del_cached_iface(self, vsctl_iface):
        vsctl_iface.port().ifaces.remove(vsctl_iface)
        vsctl_iface.port = None
        del self.ifaces[vsctl_iface.iface_cfg.name]
        vsctl_iface.iface_cfg.delete()

    def invalidate_cache(self):
        if not self.cache_valid:
            return
        self._invalidate_cache()

    def populate_cache(self):
        self._populate_cache(self.idl.tables[vswitch_idl.OVSREC_TABLE_BRIDGE])

    @staticmethod
    def port_is_fake_bridge(ovsrec_port):
        tag = ovsrec_port.tag
        if isinstance(tag, list):
            if len(tag) == 0:
                tag = 0
            else:
                tag = tag[0]
        return ovsrec_port.fake_bridge and 0 <= tag <= 4095

    def _populate_cache(self, ovsrec_bridges):
        if self.cache_valid:
            return
        self.cache_valid = True
        bridges = set()
        ports = set()
        for ovsrec_bridge in ovsrec_bridges.rows.values():
            name = ovsrec_bridge.name
            if name in bridges:
                LOG.warning('%s: database contains duplicate bridge name', name)
            bridges.add(name)
            vsctl_bridge = self.add_bridge_to_cache(ovsrec_bridge, name, None, 0)
            if not vsctl_bridge:
                continue
            for ovsrec_port in ovsrec_bridge.ports:
                port_name = ovsrec_port.name
                if port_name in ports:
                    continue
                ports.add(port_name)
                if self.port_is_fake_bridge(ovsrec_port) and port_name not in bridges:
                    bridges.add(port_name)
                    self.add_bridge_to_cache(None, port_name, vsctl_bridge, ovsrec_port.tag)
        bridges = set()
        for ovsrec_bridge in ovsrec_bridges.rows.values():
            name = ovsrec_bridge.name
            if name in bridges:
                continue
            bridges.add(name)
            vsctl_bridge = self.bridges[name]
            for ovsrec_port in ovsrec_bridge.ports:
                port_name = ovsrec_port.name
                vsctl_port = self.ports.get(port_name)
                if vsctl_port:
                    if ovsrec_port == vsctl_port.port_cfg:
                        LOG.warning('%s: vsctl_port is in multiple bridges (%s and %s)', port_name, vsctl_bridge.name, vsctl_port.br.name)
                    else:
                        LOG.error('%s: database contains duplicate vsctl_port name', ovsrec_port.name)
                    continue
                if self.port_is_fake_bridge(ovsrec_port) and port_name in bridges:
                    continue
                vsctl_port = self.add_port_to_cache(vsctl_bridge, ovsrec_port)
                for ovsrec_iface in ovsrec_port.interfaces:
                    iface = self.ifaces.get(ovsrec_iface.name)
                    if iface:
                        if ovsrec_iface == iface.iface_cfg:
                            LOG.warning('%s: interface is in multiple ports (%s and %s)', ovsrec_iface.name, iface.port().port_cfg.name, vsctl_port.port_cfg.name)
                        else:
                            LOG.error('%s: database contains duplicate interface name', ovsrec_iface.name)
                        continue
                    self.add_iface_to_cache(vsctl_port, ovsrec_iface)
                ovsrec_qos = ovsrec_port.qos
                vsctl_qos = self.add_qos_to_cache(vsctl_port, ovsrec_qos)
                if len(ovsrec_qos):
                    for ovsrec_queue in ovsrec_qos[0].queues:
                        self.add_queue_to_cache(vsctl_qos, ovsrec_queue)

    def check_conflicts(self, name, msg):
        self.verify_ports()
        if name in self.bridges:
            vsctl_fatal('%s because a bridge named %s already exists' % (msg, name))
        if name in self.ports:
            vsctl_fatal('%s because a port named %s already exists on bridge %s' % (msg, name, self.ports[name].bridge().name))
        if name in self.ifaces:
            vsctl_fatal('%s because an interface named %s already exists on bridge %s' % (msg, name, self.ifaces[name].port().bridge().name))

    def find_bridge(self, name, must_exist):
        assert self.cache_valid
        vsctl_bridge = self.bridges.get(name)
        if must_exist and (not vsctl_bridge):
            vsctl_fatal('no bridge named %s' % name)
        self.verify_bridges()
        return vsctl_bridge

    def find_real_bridge(self, name, must_exist):
        vsctl_bridge = self.find_bridge(name, must_exist)
        if vsctl_bridge and vsctl_bridge.parent:
            vsctl_fatal('%s is a fake bridge' % name)
        return vsctl_bridge

    def find_bridge_by_id(self, datapath_id, must_exist):
        assert self.cache_valid
        for vsctl_bridge in self.bridges.values():
            if vsctl_bridge.br_cfg.datapath_id[0].strip('"') == datapath_id:
                self.verify_bridges()
                return vsctl_bridge
        if must_exist:
            vsctl_fatal('no bridge id %s' % datapath_id)
        return None

    def find_port(self, name, must_exist):
        assert self.cache_valid
        vsctl_port = self.ports.get(name)
        if vsctl_port and name == vsctl_port.bridge().name:
            vsctl_port = None
        if must_exist and (not vsctl_port):
            vsctl_fatal('no vsctl_port named %s' % name)
        return vsctl_port

    def find_iface(self, name, must_exist):
        assert self.cache_valid
        vsctl_iface = self.ifaces.get(name)
        if vsctl_iface and name == vsctl_iface.port().bridge().name:
            vsctl_iface = None
        if must_exist and (not vsctl_iface):
            vsctl_fatal('no interface named %s' % name)
        self.verify_ports()
        return vsctl_iface

    def set_qos(self, vsctl_port, type, max_rate):
        qos = vsctl_port.qos.qos_cfg
        if not len(qos):
            ovsrec_qos = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_QOS])
            vsctl_port.port_cfg.qos = [ovsrec_qos]
        else:
            ovsrec_qos = qos[0]
        ovsrec_qos.type = type
        if max_rate is not None:
            value_json = ['map', [['max-rate', max_rate]]]
            self.set_column(ovsrec_qos, 'other_config', value_json)
        self.add_qos_to_cache(vsctl_port, [ovsrec_qos])
        return ovsrec_qos

    def set_queue(self, vsctl_qos, max_rate, min_rate, queue_id):
        ovsrec_qos = vsctl_qos.qos_cfg[0]
        try:
            ovsrec_queue = ovsrec_qos.queues[queue_id]
        except (AttributeError, KeyError):
            ovsrec_queue = self.txn.insert(self.txn.idl.tables[vswitch_idl.OVSREC_TABLE_QUEUE])
        if max_rate is not None:
            value_json = ['map', [['max-rate', max_rate]]]
            self.add_column(ovsrec_queue, 'other_config', value_json)
        if min_rate is not None:
            value_json = ['map', [['min-rate', min_rate]]]
            self.add_column(ovsrec_queue, 'other_config', value_json)
        value_json = ['map', [[queue_id, ['uuid', str(ovsrec_queue.uuid)]]]]
        self.add_column(ovsrec_qos, 'queues', value_json)
        self.add_queue_to_cache(vsctl_qos, ovsrec_queue)
        return ovsrec_queue

    @staticmethod
    def _column_set(ovsrec_row, column, ovsrec_value):
        setattr(ovsrec_row, column, ovsrec_value)

    @staticmethod
    def _column_insert(ovsrec_row, column, ovsrec_add):
        value = getattr(ovsrec_row, column)
        value.append(ovsrec_add)
        VSCtlContext._column_set(ovsrec_row, column, value)

    @staticmethod
    def _column_delete(ovsrec_row, column, ovsrec_del):
        value = getattr(ovsrec_row, column)
        try:
            value.remove(ovsrec_del)
        except ValueError:
            pass
        VSCtlContext._column_set(ovsrec_row, column, value)

    @staticmethod
    def bridge_insert_port(ovsrec_bridge, ovsrec_port):
        VSCtlContext._column_insert(ovsrec_bridge, vswitch_idl.OVSREC_BRIDGE_COL_PORTS, ovsrec_port)

    @staticmethod
    def bridge_delete_port(ovsrec_bridge, ovsrec_port):
        VSCtlContext._column_delete(ovsrec_bridge, vswitch_idl.OVSREC_BRIDGE_COL_PORTS, ovsrec_port)

    @staticmethod
    def port_delete_qos(ovsrec_port, ovsrec_qos):
        VSCtlContext._column_delete(ovsrec_port, vswitch_idl.OVSREC_PORT_COL_QOS, ovsrec_qos)

    def ovs_insert_bridge(self, ovsrec_bridge):
        self._column_insert(self.ovs, vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES, ovsrec_bridge)

    def ovs_delete_bridge(self, ovsrec_bridge):
        self._column_delete(self.ovs, vswitch_idl.OVSREC_OPEN_VSWITCH_COL_BRIDGES, ovsrec_bridge)

    def del_port(self, vsctl_port):
        if vsctl_port.bridge().parent:
            ovsrec_bridge = vsctl_port.bridge().parent.br_cfg
        else:
            ovsrec_bridge = vsctl_port.bridge().br_cfg
        self.bridge_delete_port(ovsrec_bridge, vsctl_port.port_cfg)
        for vsctl_iface in vsctl_port.ifaces.copy():
            self.del_cached_iface(vsctl_iface)
        self.del_cached_port(vsctl_port)

    def del_bridge(self, vsctl_bridge):
        for child in vsctl_bridge.children.copy():
            self.del_bridge(child)
        for vsctl_port in vsctl_bridge.ports.copy():
            self.del_port(vsctl_port)
        self.del_cached_bridge(vsctl_bridge)

    def del_qos(self, vsctl_qos):
        ovsrec_port = vsctl_qos.port().port_cfg
        ovsrec_qos = vsctl_qos.qos_cfg
        if len(ovsrec_qos):
            self.port_delete_qos(ovsrec_port, ovsrec_qos[0])
            self.del_cached_qos(vsctl_qos)

    def add_port(self, br_name, port_name, may_exist, fake_iface, iface_names, settings=None):
        """
        :type settings: list of (column, value_json)
                                where column is str,
                                      value_json is json that is represented
                                      by Datum.to_json()
        """
        settings = settings or []
        self.populate_cache()
        if may_exist:
            vsctl_port = self.find_port(port_name, False)
            if vsctl_port:
                want_names = set(iface_names)
                have_names = set((ovsrec_iface.name for ovsrec_iface in vsctl_port.port_cfg.interfaces))
                if vsctl_port.bridge().name != br_name:
                    vsctl_fatal('"%s" but %s is actually attached to vsctl_bridge %s' % (br_name, port_name, vsctl_port.bridge().name))
                if want_names != have_names:
                    want_names_string = ','.join(want_names)
                    have_names_string = ','.join(have_names)
                    vsctl_fatal('"%s" but %s actually has interface(s) %s' % (want_names_string, port_name, have_names_string))
                return
        self.check_conflicts(port_name, 'cannot create a port named %s' % port_name)
        for iface_name in iface_names:
            self.check_conflicts(iface_name, 'cannot create an interface named %s' % iface_name)
        vsctl_bridge = self.find_bridge(br_name, True)
        ifaces = []
        for iface_name in iface_names:
            ovsrec_iface = self.txn.insert(self.idl.tables[vswitch_idl.OVSREC_TABLE_INTERFACE])
            ovsrec_iface.name = iface_name
            ifaces.append(ovsrec_iface)
        ovsrec_port = self.txn.insert(self.idl.tables[vswitch_idl.OVSREC_TABLE_PORT])
        ovsrec_port.name = port_name
        ovsrec_port.interfaces = ifaces
        ovsrec_port.bond_fake_iface = fake_iface
        if vsctl_bridge.parent:
            tag = vsctl_bridge.vlan
            ovsrec_port.tag = tag
        for column, value in settings:
            self.set_column(ovsrec_port, column, value)
        if vsctl_bridge.parent:
            ovsrec_bridge = vsctl_bridge.parent.br_cfg
        else:
            ovsrec_bridge = vsctl_bridge.br_cfg
        self.bridge_insert_port(ovsrec_bridge, ovsrec_port)
        vsctl_port = self.add_port_to_cache(vsctl_bridge, ovsrec_port)
        for ovsrec_iface in ifaces:
            self.add_iface_to_cache(vsctl_port, ovsrec_iface)

    def add_bridge(self, br_name, parent_name=None, vlan=0, may_exist=False):
        self.populate_cache()
        if may_exist:
            vsctl_bridge = self.find_bridge(br_name, False)
            if vsctl_bridge:
                if not parent_name:
                    if vsctl_bridge.parent:
                        vsctl_fatal('"--may-exist add-vsctl_bridge %s" but %s is a VLAN bridge for VLAN %d' % (br_name, br_name, vsctl_bridge.vlan))
                elif not vsctl_bridge.parent:
                    vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s is not a VLAN bridge' % (br_name, parent_name, vlan, br_name))
                elif vsctl_bridge.parent.name != parent_name:
                    vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s has the wrong parent %s' % (br_name, parent_name, vlan, br_name, vsctl_bridge.parent.name))
                elif vsctl_bridge.vlan != vlan:
                    vsctl_fatal('"--may-exist add-vsctl_bridge %s %s %d" but %s is a VLAN bridge for the wrong VLAN %d' % (br_name, parent_name, vlan, br_name, vsctl_bridge.vlan))
                return
        self.check_conflicts(br_name, 'cannot create a bridge named %s' % br_name)
        txn = self.txn
        tables = self.idl.tables
        if not parent_name:
            ovsrec_iface = txn.insert(tables[vswitch_idl.OVSREC_TABLE_INTERFACE])
            ovsrec_iface.name = br_name
            ovsrec_iface.type = 'internal'
            ovsrec_port = txn.insert(tables[vswitch_idl.OVSREC_TABLE_PORT])
            ovsrec_port.name = br_name
            ovsrec_port.interfaces = [ovsrec_iface]
            ovsrec_port.fake_bridge = False
            ovsrec_bridge = txn.insert(tables[vswitch_idl.OVSREC_TABLE_BRIDGE])
            ovsrec_bridge.name = br_name
            ovsrec_bridge.ports = [ovsrec_port]
            self.ovs_insert_bridge(ovsrec_bridge)
        else:
            parent = self.find_bridge(parent_name, False)
            if parent and parent.parent:
                vsctl_fatal('cannot create bridge with fake bridge as parent')
            if not parent:
                vsctl_fatal('parent bridge %s does not exist' % parent_name)
            ovsrec_iface = txn.insert(tables[vswitch_idl.OVSREC_TABLE_INTERFACE])
            ovsrec_iface.name = br_name
            ovsrec_iface.type = 'internal'
            ovsrec_port = txn.insert(tables[vswitch_idl.OVSREC_TABLE_PORT])
            ovsrec_port.name = br_name
            ovsrec_port.interfaces = [ovsrec_iface]
            ovsrec_port.fake_bridge = True
            ovsrec_port.tag = vlan
            self.bridge_insert_port(parent.br_cfg, ovsrec_port)
        self.invalidate_cache()

    @staticmethod
    def parse_column_key(setting_string):
        """
        Parses 'setting_string' as str formatted in <column>[:<key>]
        and returns str type 'column' and 'key'
        """
        if ':' in setting_string:
            column, key = setting_string.split(':', 1)
        else:
            column = setting_string
            key = None
        return (column, key)

    @staticmethod
    def parse_column_key_value(table_schema, setting_string):
        """
        Parses 'setting_string' as str formatted in <column>[:<key>]=<value>
        and returns str type 'column' and json formatted 'value'
        """
        if ':' in setting_string:
            column, value = setting_string.split(':', 1)
        elif '=' in setting_string:
            column, value = setting_string.split('=', 1)
        else:
            column = setting_string
            value = None
        if value is not None:
            type_ = table_schema.columns[column].type
            value = datum_from_string(type_, value)
        return (column, value)

    def get_column(self, ovsrec_row, column, key=None, if_exists=False):
        value = getattr(ovsrec_row, column, None)
        if isinstance(value, dict) and key is not None:
            value = value.get(key, None)
            column = '%s:%s' % (column, key)
        if value is None:
            if if_exists:
                return None
            vsctl_fatal('%s does not contain a column whose name matches "%s"' % (ovsrec_row._table.name, column))
        return value

    def _pre_mod_column(self, ovsrec_row, column, value_json):
        if column not in ovsrec_row._table.columns:
            vsctl_fatal('%s does not contain a column whose name matches "%s"' % (ovsrec_row._table.name, column))
        column_schema = ovsrec_row._table.columns[column]
        datum = ovs.db.data.Datum.from_json(column_schema.type, value_json, self.symtab)
        return datum.to_python(ovs.db.idl._uuid_to_row)

    def set_column(self, ovsrec_row, column, value_json):
        column_schema = ovsrec_row._table.columns[column]
        datum = self._pre_mod_column(ovsrec_row, column, value_json)
        if column_schema.type.is_map():
            values = getattr(ovsrec_row, column, {})
            values.update(datum)
        else:
            values = datum
        setattr(ovsrec_row, column, values)

    def add_column(self, ovsrec_row, column, value_json):
        column_schema = ovsrec_row._table.columns[column]
        datum = self._pre_mod_column(ovsrec_row, column, value_json)
        if column_schema.type.is_map():
            values = getattr(ovsrec_row, column, {})
            values.update(datum)
        elif column_schema.type.is_set():
            values = getattr(ovsrec_row, column, [])
            values.extend(datum)
        else:
            values = datum
        setattr(ovsrec_row, column, values)

    def remove_column(self, ovsrec_row, column, value_json):
        column_schema = ovsrec_row._table.columns[column]
        datum = self._pre_mod_column(ovsrec_row, column, value_json)
        if column_schema.type.is_map():
            values = getattr(ovsrec_row, column, {})
            for datum_key, datum_value in datum.items():
                v = values.get(datum_key, None)
                if v == datum_value:
                    values.pop(datum_key)
            setattr(ovsrec_row, column, values)
        elif column_schema.type.is_set():
            values = getattr(ovsrec_row, column, [])
            for d in datum:
                if d in values:
                    values.remove(d)
            setattr(ovsrec_row, column, values)
        else:
            values = getattr(ovsrec_row, column, None)
            default = ovs.db.data.Datum.default(column_schema.type)
            default = default.to_python(ovs.db.idl._uuid_to_row).to_json()
            if values == datum:
                setattr(ovsrec_row, column, default)

    def _get_row_by_id(self, table_name, vsctl_row_id, record_id):
        if not vsctl_row_id.table:
            return None
        if not vsctl_row_id.name_column:
            if record_id != '.':
                return None
            values = list(self.idl.tables[vsctl_row_id.table].rows.values())
            if not values or len(values) > 2:
                return None
            referrer = values[0]
        else:
            referrer = None
            for ovsrec_row in self.idl.tables[vsctl_row_id.table].rows.values():
                name = getattr(ovsrec_row, vsctl_row_id.name_column)
                assert isinstance(name, (list, str, str))
                if not isinstance(name, list) and name == record_id:
                    if referrer:
                        vsctl_fatal('multiple rows in %s match "%s"' % (table_name, record_id))
                    referrer = ovsrec_row
        if not referrer:
            return None
        final = None
        if vsctl_row_id.uuid_column:
            referrer.verify(vsctl_row_id.uuid_column)
            uuid = getattr(referrer, vsctl_row_id.uuid_column)
            uuid_ = referrer._data[vsctl_row_id.uuid_column]
            assert uuid_.type.key.type == ovs.db.types.UuidType
            assert uuid_.type.value is None
            assert isinstance(uuid, list)
            if len(uuid) == 1:
                final = uuid[0]
        else:
            final = referrer
        return final

    def get_row(self, vsctl_table, record_id):
        table_name = vsctl_table.table_name
        if ovsuuid.is_valid_string(record_id):
            uuid = ovsuuid.from_string(record_id)
            return self.idl.tables[table_name].rows.get(uuid)
        else:
            for vsctl_row_id in vsctl_table.row_ids:
                ovsrec_row = self._get_row_by_id(table_name, vsctl_row_id, record_id)
                if ovsrec_row:
                    return ovsrec_row
        return None

    def must_get_row(self, vsctl_table, record_id):
        ovsrec_row = self.get_row(vsctl_table, record_id)
        if not ovsrec_row:
            vsctl_fatal('no row "%s" in table %s' % (record_id, vsctl_table.table_name))
        return ovsrec_row