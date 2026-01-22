import json
import numpy as np
from psycopg2 import connect
from psycopg2.extras import execute_values
from ase.db.sqlite import (init_statements, index_statements, VERSION,
from ase.io.jsonio import (encode as ase_encode,
class PostgreSQLDatabase(SQLite3Database):
    type = 'postgresql'
    default = 'DEFAULT'

    def encode(self, obj, binary=False):
        return ase_encode(remove_nan_and_inf(obj))

    def decode(self, obj, lazy=False):
        return insert_ase_and_ndarray_objects(insert_nan_and_inf(obj))

    def blob(self, array):
        """Convert array to blob/buffer object."""
        if array is None:
            return None
        if len(array) == 0:
            array = np.zeros(0)
        if array.dtype == np.int64:
            array = array.astype(np.int32)
        return array.tolist()

    def deblob(self, buf, dtype=float, shape=None):
        """Convert blob/buffer object to ndarray of correct dtype and shape.

        (without creating an extra view)."""
        if buf is None:
            return None
        return np.array(buf, dtype=dtype)

    def _connect(self):
        return Connection(connect(self.filename))

    def _initialize(self, con):
        if self.initialized:
            return
        self._metadata = {}
        cur = con.cursor()
        cur.execute('show search_path;')
        schema = cur.fetchone()[0].split(', ')
        if schema[0] == '"$user"':
            schema = schema[1]
        else:
            schema = schema[0]
        cur.execute("\n        SELECT EXISTS(select * from information_schema.tables where\n        table_name='information' and table_schema='{}');\n        ".format(schema))
        if not cur.fetchone()[0]:
            sql = ';\n'.join(init_statements)
            sql = schema_update(sql)
            cur.execute(sql)
            if self.create_indices:
                cur.execute(';\n'.join(index_statements))
                cur.execute(';\n'.join(jsonb_indices))
            con.commit()
            self.version = VERSION
        else:
            cur.execute('select * from information;')
            for name, value in cur.fetchall():
                if name == 'version':
                    self.version = int(value)
                elif name == 'metadata':
                    self._metadata = json.loads(value)
        assert 5 < self.version <= VERSION
        self.initialized = True

    def get_offset_string(self, offset, limit=None):
        return '\nOFFSET {0}'.format(offset)

    def get_last_id(self, cur):
        cur.execute('SELECT last_value FROM systems_id_seq')
        id = cur.fetchone()[0]
        return int(id)