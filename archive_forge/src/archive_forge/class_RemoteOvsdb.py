import collections
import errno
import uuid
from ovs import jsonrpc
from ovs import poller
from ovs import reconnect
from ovs import stream
from ovs import timeval
from ovs.db import idl
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.services.protocols.ovsdb import event
from os_ken.services.protocols.ovsdb import model
class RemoteOvsdb(app_manager.OSKenApp):
    _EVENTS = [event.EventRowUpdate, event.EventRowDelete, event.EventRowInsert, event.EventInterfaceDeleted, event.EventInterfaceInserted, event.EventInterfaceUpdated, event.EventPortDeleted, event.EventPortInserted, event.EventPortUpdated]

    @classmethod
    def factory(cls, sock, address, probe_interval=None, min_backoff=None, max_backoff=None, schema_tables=None, schema_exclude_columns=None, *args, **kwargs):
        schema_exclude_columns = schema_exclude_columns or {}
        ovs_stream = stream.Stream(sock, None, None)
        connection = jsonrpc.Connection(ovs_stream)
        schemas = discover_schemas(connection)
        if not schemas:
            return
        if schema_tables or schema_exclude_columns:
            schemas = _filter_schemas(schemas, schema_tables, schema_exclude_columns)
        fsm = reconnect.Reconnect(now())
        fsm.set_name('%s:%s' % address[:2])
        fsm.enable(now())
        fsm.set_passive(True, now())
        fsm.set_max_tries(-1)
        if probe_interval is not None:
            fsm.set_probe_interval(probe_interval)
        if min_backoff is None:
            min_backoff = fsm.get_min_backoff()
        if max_backoff is None:
            max_backoff = fsm.get_max_backoff()
        if min_backoff and max_backoff:
            fsm.set_backoff(min_backoff, max_backoff)
        fsm.connected(now())
        session = jsonrpc.Session(fsm, connection, fsm.get_name())
        idl = Idl(session, schemas[0])
        system_id = discover_system_id(idl)
        if not system_id:
            return None
        name = cls.instance_name(system_id)
        ovs_stream.name = name
        connection.name = name
        fsm.set_name(name)
        kwargs = kwargs.copy()
        kwargs['socket'] = sock
        kwargs['address'] = address
        kwargs['idl'] = idl
        kwargs['name'] = name
        kwargs['system_id'] = system_id
        app_mgr = app_manager.AppManager.get_instance()
        old_app = app_manager.lookup_service_brick(name)
        old_events = None
        if old_app:
            old_events = old_app.events
            app_mgr.uninstantiate(name)
        app = app_mgr.instantiate(cls, *args, **kwargs)
        if old_events:
            app.events = old_events
        return app

    @classmethod
    def instance_name(cls, system_id):
        return '%s-%s' % (cls.__name__, system_id)

    def __init__(self, *args, **kwargs):
        super(RemoteOvsdb, self).__init__(*args, **kwargs)
        self.socket = kwargs['socket']
        self.address = kwargs['address']
        self._idl = kwargs['idl']
        self.system_id = kwargs['system_id']
        self.name = kwargs['name']
        self._txn_q = collections.deque()

    def _event_proxy_loop(self):
        while self.is_active:
            events = self._idl.events
            if not events:
                hub.sleep(0.1)
                continue
            for e in events:
                ev = e[0]
                args = e[1]
                self._submit_event(ev(self.system_id, *args))
            hub.sleep(0)

    def _submit_event(self, ev):
        self.send_event_to_observers(ev)
        try:
            ev_cls_name = 'Event' + ev.table + ev.event_type
            proxy_ev_cls = getattr(event, ev_cls_name, None)
            if proxy_ev_cls:
                self.send_event_to_observers(proxy_ev_cls(ev))
        except Exception:
            self.logger.exception('Error submitting specific event for OVSDB %s', self.system_id)

    def _idl_loop(self):
        while self.is_active:
            try:
                self._idl.run()
                self._transactions()
            except Exception:
                self.logger.exception('Error running IDL for system_id %s' % self.system_id)
                raise
            hub.sleep(0)

    def _run_thread(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            self.stop()

    def _transactions(self):
        if not self._txn_q:
            return
        self._transaction()

    def _transaction(self):
        req = self._txn_q.popleft()
        txn = idl.Transaction(self._idl)
        uuids = req.func(self._idl.tables, txn.insert)
        status = txn.commit_block()
        insert_uuids = {}
        err_msg = None
        if status in (idl.Transaction.SUCCESS, idl.Transaction.UNCHANGED):
            if uuids:
                if isinstance(uuids, uuid.UUID):
                    insert_uuids[uuids] = txn.get_insert_uuid(uuids)
                else:
                    insert_uuids = dict(((uuid, txn.get_insert_uuid(uuid)) for uuid in uuids))
        else:
            err_msg = txn.get_error()
        rep = event.EventModifyReply(self.system_id, status, insert_uuids, err_msg)
        self.reply_to_request(req, rep)

    def modify_request_handler(self, ev):
        self._txn_q.append(ev)

    def read_request_handler(self, ev, bulk=False):
        result = ev.func(self._idl.tables)
        if bulk:
            return (self.system_id, result)
        rep = event.EventReadReply(self.system_id, result)
        self.reply_to_request(ev, rep)

    def start(self):
        super(RemoteOvsdb, self).start()
        t = hub.spawn(self._run_thread, self._idl_loop)
        self.threads.append(t)
        t = hub.spawn(self._run_thread, self._event_proxy_loop)
        self.threads.append(t)

    def stop(self):
        self.is_active = False
        hub.joinall(self.threads)
        self._idl.close()
        super(RemoteOvsdb, self).stop()