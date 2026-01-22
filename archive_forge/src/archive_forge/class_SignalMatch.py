import logging
import threading
import weakref
from _dbus_bindings import (
from dbus.exceptions import DBusException
from dbus.lowlevel import (
from dbus.proxies import ProxyObject
from dbus._compat import is_py2, is_py3
from _dbus_bindings import String
class SignalMatch(object):
    _slots = ['_sender_name_owner', '_member', '_interface', '_sender', '_path', '_handler', '_args_match', '_rule', '_byte_arrays', '_conn_weakref', '_destination_keyword', '_interface_keyword', '_message_keyword', '_member_keyword', '_sender_keyword', '_path_keyword', '_int_args_match']
    __slots__ = tuple(_slots)

    def __init__(self, conn, sender, object_path, dbus_interface, member, handler, byte_arrays=False, sender_keyword=None, path_keyword=None, interface_keyword=None, member_keyword=None, message_keyword=None, destination_keyword=None, **kwargs):
        if member is not None:
            validate_member_name(member)
        if dbus_interface is not None:
            validate_interface_name(dbus_interface)
        if sender is not None:
            validate_bus_name(sender)
        if object_path is not None:
            validate_object_path(object_path)
        self._rule = None
        self._conn_weakref = weakref.ref(conn)
        self._sender = sender
        self._interface = dbus_interface
        self._member = member
        self._path = object_path
        self._handler = handler
        self._sender_name_owner = sender
        if 'utf8_strings' in kwargs:
            raise TypeError("unexpected keyword argument 'utf8_strings'")
        self._byte_arrays = byte_arrays
        self._sender_keyword = sender_keyword
        self._path_keyword = path_keyword
        self._member_keyword = member_keyword
        self._interface_keyword = interface_keyword
        self._message_keyword = message_keyword
        self._destination_keyword = destination_keyword
        self._args_match = kwargs
        if not kwargs:
            self._int_args_match = None
        else:
            self._int_args_match = {}
            for kwarg in kwargs:
                if not kwarg.startswith('arg'):
                    raise TypeError('SignalMatch: unknown keyword argument %s' % kwarg)
                try:
                    index = int(kwarg[3:])
                except ValueError:
                    raise TypeError('SignalMatch: unknown keyword argument %s' % kwarg)
                if index < 0 or index > 63:
                    raise TypeError('SignalMatch: arg match index must be in range(64), not %d' % index)
                self._int_args_match[index] = kwargs[kwarg]

    def __hash__(self):
        """SignalMatch objects are compared by identity."""
        return hash(id(self))

    def __eq__(self, other):
        """SignalMatch objects are compared by identity."""
        return self is other

    def __ne__(self, other):
        """SignalMatch objects are compared by identity."""
        return self is not other
    sender = property(lambda self: self._sender)

    def __str__(self):
        if self._rule is None:
            rule = ["type='signal'"]
            if self._sender is not None:
                rule.append("sender='%s'" % self._sender)
            if self._path is not None:
                rule.append("path='%s'" % self._path)
            if self._interface is not None:
                rule.append("interface='%s'" % self._interface)
            if self._member is not None:
                rule.append("member='%s'" % self._member)
            if self._int_args_match is not None:
                for index, value in self._int_args_match.items():
                    rule.append("arg%d='%s'" % (index, value))
            self._rule = ','.join(rule)
        return self._rule

    def __repr__(self):
        return '<%s at %x "%s" on conn %r>' % (self.__class__, id(self), self._rule, self._conn_weakref())

    def set_sender_name_owner(self, new_name):
        self._sender_name_owner = new_name

    def matches_removal_spec(self, sender, object_path, dbus_interface, member, handler, **kwargs):
        if handler not in (None, self._handler):
            return False
        if sender != self._sender:
            return False
        if object_path != self._path:
            return False
        if dbus_interface != self._interface:
            return False
        if member != self._member:
            return False
        if kwargs != self._args_match:
            return False
        return True

    def maybe_handle_message(self, message):
        args = None
        if self._sender_name_owner not in (None, message.get_sender()):
            return False
        if self._int_args_match is not None:
            kwargs = dict(byte_arrays=True)
            args = message.get_args_list(**kwargs)
            for index, value in self._int_args_match.items():
                if index >= len(args) or not isinstance(args[index], String) or args[index] != value:
                    return False
        if self._member not in (None, message.get_member()):
            return False
        if self._interface not in (None, message.get_interface()):
            return False
        if self._path not in (None, message.get_path()):
            return False
        try:
            if args is None or not self._byte_arrays:
                args = message.get_args_list(byte_arrays=self._byte_arrays)
            kwargs = {}
            if self._sender_keyword is not None:
                kwargs[self._sender_keyword] = message.get_sender()
            if self._destination_keyword is not None:
                kwargs[self._destination_keyword] = message.get_destination()
            if self._path_keyword is not None:
                kwargs[self._path_keyword] = message.get_path()
            if self._member_keyword is not None:
                kwargs[self._member_keyword] = message.get_member()
            if self._interface_keyword is not None:
                kwargs[self._interface_keyword] = message.get_interface()
            if self._message_keyword is not None:
                kwargs[self._message_keyword] = message
            self._handler(*args, **kwargs)
        except:
            logging.basicConfig()
            _logger.error('Exception in handler for D-Bus signal:', exc_info=1)
        return True

    def remove(self):
        conn = self._conn_weakref()
        if conn is not None:
            conn.remove_signal_receiver(self, self._member, self._interface, self._sender, self._path, **self._args_match)