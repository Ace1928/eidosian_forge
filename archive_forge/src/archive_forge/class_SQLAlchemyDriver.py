import logging
from oslo_serialization import jsonutils
from osprofiler.drivers import base
from osprofiler import exc
class SQLAlchemyDriver(base.Driver):

    def __init__(self, connection_str, project=None, service=None, host=None, **kwargs):
        super(SQLAlchemyDriver, self).__init__(connection_str, project=project, service=service, host=host)
        try:
            from sqlalchemy import create_engine
            from sqlalchemy import Table, MetaData, Column
            from sqlalchemy import String, JSON, Integer
        except ImportError:
            LOG.exception("To use this command, install 'SQLAlchemy'")
        else:
            self._metadata = MetaData()
            self._data_table = Table('data', self._metadata, Column('id', Integer, primary_key=True), Column('timestamp', String(26), index=True), Column('base_id', String(255), index=True), Column('parent_id', String(255), index=True), Column('trace_id', String(255), index=True), Column('project', String(255), index=True), Column('host', String(255), index=True), Column('service', String(255), index=True), Column('name', String(255), index=True), Column('data', JSON))
        try:
            self._engine = create_engine(connection_str)
            self._conn = self._engine.connect()
            self._metadata.create_all(self._engine, checkfirst=True)
        except Exception:
            LOG.exception('Failed to create engine/connection and setup intial database tables')

    @classmethod
    def get_name(cls):
        return 'sqlalchemy'

    def notify(self, info, context=None):
        """Write a notification the the database"""
        data = info.copy()
        base_id = data.pop('base_id', None)
        timestamp = data.pop('timestamp', None)
        parent_id = data.pop('parent_id', None)
        trace_id = data.pop('trace_id', None)
        project = data.pop('project', self.project)
        host = data.pop('host', self.host)
        service = data.pop('service', self.service)
        name = data.pop('name', None)
        try:
            ins = self._data_table.insert().values(timestamp=timestamp, base_id=base_id, parent_id=parent_id, trace_id=trace_id, project=project, service=service, host=host, name=name, data=jsonutils.dumps(data))
            self._conn.execute(ins)
        except Exception:
            LOG.exception('Can not store osprofiler tracepoint {} (base_id {})'.format(trace_id, base_id))

    def list_traces(self, fields=None):
        try:
            from sqlalchemy.sql import select
        except ImportError:
            raise exc.CommandError("To use this command, you should install 'SQLAlchemy'")
        stmt = select([self._data_table])
        seen_ids = set()
        result = []
        traces = self._conn.execute(stmt).fetchall()
        for trace in traces:
            if trace['base_id'] not in seen_ids:
                seen_ids.add(trace['base_id'])
                result.append({key: value for key, value in trace.items() if key in fields})
        return result

    def get_report(self, base_id):
        try:
            from sqlalchemy.sql import select
        except ImportError:
            raise exc.CommandError("To use this command, you should install 'SQLAlchemy'")
        stmt = select([self._data_table]).where(self._data_table.c.base_id == base_id)
        results = self._conn.execute(stmt).fetchall()
        for n in results:
            timestamp = n['timestamp']
            trace_id = n['trace_id']
            parent_id = n['parent_id']
            name = n['name']
            project = n['project']
            service = n['service']
            host = n['host']
            data = jsonutils.loads(n['data'])
            self._append_results(trace_id, parent_id, name, project, service, host, timestamp, data)
        return self._parse_results()