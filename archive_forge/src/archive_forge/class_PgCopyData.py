from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.database import (
from ansible_collections.community.postgresql.plugins.module_utils.postgres import (
class PgCopyData(object):
    """Implements behavior of COPY FROM, COPY TO PostgreSQL command.

    Arguments:
        module (AnsibleModule) -- object of AnsibleModule class
        cursor (cursor) -- cursor object of psycopg library

    Attributes:
        module (AnsibleModule) -- object of AnsibleModule class
        cursor (cursor) -- cursor object of psycopg library
        changed (bool) --  something was changed after execution or not
        executed_queries (list) -- executed queries
        dst (str) -- data destination table (when copy_from)
        src (str) -- data source table (when copy_to)
        opt_need_quotes (tuple) -- values of these options must be passed
            to SQL in quotes
    """

    def __init__(self, module, cursor):
        self.module = module
        self.cursor = cursor
        self.executed_queries = []
        self.changed = False
        self.dst = ''
        self.src = ''
        self.opt_need_quotes = ('DELIMITER', 'NULL', 'QUOTE', 'ESCAPE', 'ENCODING')

    def copy_from(self):
        """Implements COPY FROM command behavior."""
        self.src = self.module.params['copy_from']
        self.dst = self.module.params['dst']
        query_fragments = ['COPY %s' % pg_quote_identifier(self.dst, 'table')]
        if self.module.params.get('columns'):
            query_fragments.append('(%s)' % ','.join(self.module.params['columns']))
        query_fragments.append('FROM')
        if self.module.params.get('program'):
            query_fragments.append('PROGRAM')
        query_fragments.append("'%s'" % self.src)
        if self.module.params.get('options'):
            query_fragments.append(self.__transform_options())
        if self.module.check_mode:
            self.changed = self.__check_table(self.dst)
            if self.changed:
                self.executed_queries.append(' '.join(query_fragments))
        elif exec_sql(self, ' '.join(query_fragments), return_bool=True):
            self.changed = True

    def copy_to(self):
        """Implements COPY TO command behavior."""
        self.src = self.module.params['src']
        self.dst = self.module.params['copy_to']
        if 'SELECT ' in self.src.upper():
            query_fragments = ['COPY (%s)' % self.src]
        else:
            query_fragments = ['COPY %s' % pg_quote_identifier(self.src, 'table')]
        if self.module.params.get('columns'):
            query_fragments.append('(%s)' % ','.join(self.module.params['columns']))
        query_fragments.append('TO')
        if self.module.params.get('program'):
            query_fragments.append('PROGRAM')
        query_fragments.append("'%s'" % self.dst)
        if self.module.params.get('options'):
            query_fragments.append(self.__transform_options())
        if self.module.check_mode:
            self.changed = self.__check_table(self.src)
            if self.changed:
                self.executed_queries.append(' '.join(query_fragments))
        elif exec_sql(self, ' '.join(query_fragments), return_bool=True):
            self.changed = True

    def __transform_options(self):
        """Transform options dict into a suitable string."""
        for key, val in iteritems(self.module.params['options']):
            if key.upper() in self.opt_need_quotes:
                self.module.params['options'][key] = "'%s'" % val
        opt = ['%s %s' % (key, val) for key, val in iteritems(self.module.params['options'])]
        return '(%s)' % ', '.join(opt)

    def __check_table(self, table):
        """Check table or SQL in transaction mode for check_mode.

        Return True if it is OK.

        Arguments:
            table (str) - Table name that needs to be checked.
                It can be SQL SELECT statement that was passed
                instead of the table name.
        """
        if 'SELECT ' in table.upper():
            exec_sql(self, table, add_to_executed=False)
            return True
        exec_sql(self, 'SELECT 1 FROM %s' % pg_quote_identifier(table, 'table'), add_to_executed=False)
        return True