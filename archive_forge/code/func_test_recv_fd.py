import pytest
from jeepney import new_method_call, MessageType, DBusAddress
from jeepney.bus_messages import message_bus, MatchRule
from jeepney.io.blocking import open_dbus_connection, Proxy
from .utils import have_session_bus
def test_recv_fd(respond_with_fd):
    getfd_call = new_method_call(respond_with_fd, 'GetFD')
    with open_dbus_connection(bus='SESSION', enable_fds=True) as conn:
        reply = conn.send_and_get_reply(getfd_call, timeout=5)
    assert reply.header.message_type is MessageType.method_return
    with reply.body[0].to_file('w+') as f:
        assert f.read() == 'readme'