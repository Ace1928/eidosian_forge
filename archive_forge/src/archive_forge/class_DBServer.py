import os
from . import BioSeq
from . import Loader
from . import DBUtils
class DBServer:
    """Represents a BioSQL database containing namespaces (sub-databases).

    This acts like a Python dictionary, giving access to each namespace
    (defined by a row in the biodatabase table) as a BioSeqDatabase object.
    """

    def __init__(self, conn, module, module_name=None):
        """Create a DBServer object.

        Arguments:
         - conn - A database connection object
         - module - The module used to create the database connection
         - module_name - Optionally, the name of the module. Default: module.__name__

        Normally you would not want to create a DBServer object yourself.
        Instead use the open_database function, which returns an instance of DBServer.
        """
        self.module = module
        if module_name is None:
            module_name = module.__name__
        if module_name == 'mysql.connector':
            wrap_cursor = True
        else:
            wrap_cursor = False
        Adapt = _interface_specific_adaptors.get(module_name, Adaptor)
        self.adaptor = Adapt(conn, DBUtils.get_dbutils(module_name), wrap_cursor=wrap_cursor)
        self.module_name = module_name

    def __repr__(self):
        """Return a short description of the class name and database connection."""
        return f'{self.__class__.__name__}({self.adaptor.conn!r})'

    def __getitem__(self, name):
        """Return a BioSeqDatabase object.

        Arguments:
            - name - The name of the BioSeqDatabase

        """
        return BioSeqDatabase(self.adaptor, name)

    def __len__(self):
        """Return number of namespaces (sub-databases) in this database."""
        sql = 'SELECT COUNT(name) FROM biodatabase;'
        return int(self.adaptor.execute_and_fetch_col0(sql)[0])

    def __contains__(self, value):
        """Check if a namespace (sub-database) in this database."""
        sql = 'SELECT COUNT(name) FROM biodatabase WHERE name=%s;'
        return bool(self.adaptor.execute_and_fetch_col0(sql, (value,))[0])

    def __iter__(self):
        """Iterate over namespaces (sub-databases) in the database."""
        return iter(self.adaptor.list_biodatabase_names())

    def keys(self):
        """Iterate over namespaces (sub-databases) in the database."""
        return iter(self)

    def values(self):
        """Iterate over BioSeqDatabase objects in the database."""
        for key in self:
            yield self[key]

    def items(self):
        """Iterate over (namespace, BioSeqDatabase) in the database."""
        for key in self:
            yield (key, self[key])

    def __delitem__(self, name):
        """Remove a namespace and all its entries."""
        if name not in self:
            raise KeyError(name)
        db_id = self.adaptor.fetch_dbid_by_dbname(name)
        remover = Loader.DatabaseRemover(self.adaptor, db_id)
        remover.remove()

    def new_database(self, db_name, authority=None, description=None):
        """Add a new database to the server and return it."""
        sql = 'INSERT INTO biodatabase (name, authority, description) VALUES (%s, %s, %s)'
        self.adaptor.execute(sql, (db_name, authority, description))
        return BioSeqDatabase(self.adaptor, db_name)

    def load_database_sql(self, sql_file):
        """Load a database schema into the given database.

        This is used to create tables, etc when a database is first created.
        sql_file should specify the complete path to a file containing
        SQL entries for building the tables.
        """
        sql = ''
        with open(sql_file) as sql_handle:
            for line in sql_handle:
                if line.startswith('--'):
                    pass
                elif line.startswith('#'):
                    pass
                elif line.strip():
                    sql += line.strip() + ' '
        if self.module_name in ['psycopg2', 'pgdb']:
            self.adaptor.cursor.execute(sql)
        elif self.module_name in ['mysql.connector', 'MySQLdb', 'sqlite3']:
            sql_parts = sql.split(';')
            for sql_line in sql_parts[:-1]:
                self.adaptor.cursor.execute(sql_line)
        else:
            raise ValueError(f'Module {self.module_name} not supported by the loader.')

    def commit(self):
        """Commit the current transaction to the database."""
        return self.adaptor.commit()

    def rollback(self):
        """Roll-back the current transaction."""
        return self.adaptor.rollback()

    def close(self):
        """Close the connection. No further activity possible."""
        return self.adaptor.close()