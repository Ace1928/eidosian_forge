from tempfile import TemporaryFile
import threading
import pytest
from jeepney import (
from jeepney.io.threading import open_dbus_connection, DBusRouter, Proxy
def _reply_once():
    while True:
        msg = conn.receive()
        if msg.header.message_type is MessageType.method_call:
            if msg.header.fields[HeaderFields.member] == 'ReadFD':
                with msg.body[0].to_file('rb') as f:
                    f.seek(0)
                    b = f.read()
                conn.send(new_method_return(msg, 'ay', (b,)))
                return
            else:
                conn.send(new_error(msg, 'NoMethod'))