import os
from . import BioSeq
from . import Loader
from . import DBUtils
class MysqlConnectorAdaptor(Adaptor):
    """A BioSQL Adaptor class with fixes for the MySQL interface.

    BioSQL was failing due to returns of bytearray objects from
    the mysql-connector-python database connector. This adaptor
    class scrubs returns of bytearrays and of byte strings converting
    them to string objects instead. This adaptor class was made in
    response to backwards incompatible changes added to
    mysql-connector-python in release 2.0.0 of the package.
    """

    @staticmethod
    def _bytearray_to_str(s):
        """If s is bytes or bytearray, convert to a string (PRIVATE)."""
        if isinstance(s, (bytes, bytearray)):
            return s.decode()
        return s

    def execute_one(self, sql, args=None):
        """Execute sql that returns 1 record, and return the record."""
        out = super().execute_one(sql, args)
        return tuple((self._bytearray_to_str(v) for v in out))

    def execute_and_fetch_col0(self, sql, args=None):
        """Return a list of values from the first column in the row."""
        out = super().execute_and_fetch_col0(sql, args)
        return [self._bytearray_to_str(column) for column in out]

    def execute_and_fetchall(self, sql, args=None):
        """Return a list of tuples of all rows."""
        out = super().execute_and_fetchall(sql, args)
        return [tuple((self._bytearray_to_str(v) for v in o)) for o in out]