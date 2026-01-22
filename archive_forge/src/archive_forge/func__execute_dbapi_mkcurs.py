import datetime
import logging
from petl.compat import long, text_type
from petl.errors import ArgumentError
from petl.util.materialise import columns
from petl.transform.basics import head
from petl.io.db_utils import _is_dbapi_connection, _is_dbapi_cursor, \
def _execute_dbapi_mkcurs(sql, mkcurs, commit):
    debug('obtain a cursor')
    cursor = mkcurs()
    debug('execute SQL')
    cursor.execute(sql)
    debug('close the cursor')
    cursor.close()
    if commit:
        debug('commit transaction')
        assert hasattr(cursor, 'connection'), 'could not obtain connection via cursor'
        connection = cursor.connection
        connection.commit()