from _dbus_bindings import _Server
from dbus.connection import Connection
def _on_new_connection(self, conn):
    conn.call_on_disconnection(self.connection_removed)
    self.connection_added(conn)