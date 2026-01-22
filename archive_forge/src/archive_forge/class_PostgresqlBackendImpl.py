import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
@BackendImpl.impl.dispatch_for('postgresql')
class PostgresqlBackendImpl(BackendImpl):

    def create_opportunistic_driver_url(self):
        return 'postgresql+psycopg2://openstack_citest:openstack_citest@localhost/postgres'

    def create_named_database(self, engine, ident, conditional=False):
        with engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
            if not conditional or not self.database_exists(conn, ident):
                conn.exec_driver_sql('CREATE DATABASE %s' % ident)

    def drop_named_database(self, engine, ident, conditional=False):
        with engine.connect().execution_options(isolation_level='AUTOCOMMIT') as conn:
            self._close_out_database_users(conn, ident)
            if conditional:
                conn.exec_driver_sql('DROP DATABASE IF EXISTS %s' % ident)
            else:
                conn.exec_driver_sql('DROP DATABASE %s' % ident)

    def drop_additional_objects(self, conn):
        enums = [e['name'] for e in sqlalchemy.inspect(conn).get_enums()]
        for e in enums:
            conn.exec_driver_sql('DROP TYPE %s' % e)

    def database_exists(self, engine, ident):
        return bool(engine.scalar(sqlalchemy.text('SELECT datname FROM pg_database WHERE datname=:name'), {'name': ident}))

    def _close_out_database_users(self, conn, ident):
        """Attempt to guarantee a database can be dropped.

        Optional feature which guarantees no connections with our
        username are attached to the DB we're going to drop.

        This method has caveats; for one, the 'pid' column was named
        'procpid' prior to Postgresql 9.2.  But more critically,
        prior to 9.2 this operation required superuser permissions,
        even if the connections we're closing are under the same username
        as us.   In more recent versions this restriction has been
        lifted for same-user connections.

        """
        if conn.dialect.server_version_info >= (9, 2):
            conn.execute(sqlalchemy.text('SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename=current_user AND pid != pg_backend_pid() AND datname=:dname'), {'dname': ident})