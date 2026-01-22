from __future__ import absolute_import, division, print_function
import re
from fnmatch import fnmatch
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import \
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgDbConn(object):
    """Auxiliary class for working with PostgreSQL connection objects.

    Arguments:
        module (AnsibleModule): Object of AnsibleModule class that
            contains connection parameters.
    """

    def __init__(self, module):
        self.module = module
        self.db_conn = None
        self.cursor = None

    def connect(self, fail_on_conn=True):
        """Connect to a PostgreSQL database and return a cursor object.

        Note: connection parameters are passed by self.module object.
        """
        ensure_required_libs(self.module)
        conn_params = get_conn_params(self.module, self.module.params, warn_db_default=False)
        self.db_conn, dummy = connect_to_db(self.module, conn_params, fail_on_conn=fail_on_conn)
        if self.db_conn is None:
            return None
        return self.db_conn.cursor(**pg_cursor_args)

    def reconnect(self, dbname):
        """Reconnect to another database and return a PostgreSQL cursor object.

        Arguments:
            dbname (string): Database name to connect to.
        """
        if self.db_conn is not None:
            self.db_conn.close()
        self.module.params['db'] = dbname
        self.module.params['database'] = dbname
        self.module.params['login_db'] = dbname
        return self.connect(fail_on_conn=False)