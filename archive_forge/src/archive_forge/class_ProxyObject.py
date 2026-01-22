import logging
import _dbus_bindings
from dbus._expat_introspect_parser import process_introspection_data
from dbus.exceptions import (
from _dbus_bindings import (
from dbus._compat import is_py2
class ProxyObject(object):
    """A proxy to the remote Object.

    A ProxyObject is provided by the Bus. ProxyObjects
    have member functions, and can be called like normal Python objects.
    """
    ProxyMethodClass = _ProxyMethod
    DeferredMethodClass = _DeferredMethod
    INTROSPECT_STATE_DONT_INTROSPECT = 0
    INTROSPECT_STATE_INTROSPECT_IN_PROGRESS = 1
    INTROSPECT_STATE_INTROSPECT_DONE = 2

    def __init__(self, conn=None, bus_name=None, object_path=None, introspect=True, follow_name_owner_changes=False, **kwargs):
        """Initialize the proxy object.

        :Parameters:
            `conn` : `dbus.connection.Connection`
                The bus or connection on which to find this object.
                The keyword argument `bus` is a deprecated alias for this.
            `bus_name` : str
                A bus name for the application owning the object, to be used
                as the destination for method calls and the sender for
                signal matches. The keyword argument ``named_service`` is a
                deprecated alias for this.
            `object_path` : str
                The object path at which the application exports the object
            `introspect` : bool
                If true (default), attempt to introspect the remote
                object to find out supported methods and their signatures
            `follow_name_owner_changes` : bool
                If true (default is false) and the `bus_name` is a
                well-known name, follow ownership changes for that name
        """
        bus = kwargs.pop('bus', None)
        if bus is not None:
            if conn is not None:
                raise TypeError('conn and bus cannot both be specified')
            conn = bus
            from warnings import warn
            warn('Passing the bus parameter to ProxyObject by name is deprecated: please use positional parameters', DeprecationWarning, stacklevel=2)
        named_service = kwargs.pop('named_service', None)
        if named_service is not None:
            if bus_name is not None:
                raise TypeError('bus_name and named_service cannot both be specified')
            bus_name = named_service
            from warnings import warn
            warn('Passing the named_service parameter to ProxyObject by name is deprecated: please use positional parameters', DeprecationWarning, stacklevel=2)
        if kwargs:
            raise TypeError('ProxyObject.__init__ does not take these keyword arguments: %s' % ', '.join(kwargs.keys()))
        if follow_name_owner_changes:
            conn._require_main_loop()
        self._bus = conn
        if bus_name is not None:
            _dbus_bindings.validate_bus_name(bus_name)
        self._named_service = self._requested_bus_name = bus_name
        _dbus_bindings.validate_object_path(object_path)
        self.__dbus_object_path__ = object_path
        if not follow_name_owner_changes:
            self._named_service = conn.activate_name_owner(bus_name)
        self._pending_introspect = None
        self._pending_introspect_queue = []
        self._introspect_method_map = {}
        self._introspect_lock = RLock()
        if not introspect or self.__dbus_object_path__ == LOCAL_PATH:
            self._introspect_state = self.INTROSPECT_STATE_DONT_INTROSPECT
        else:
            self._introspect_state = self.INTROSPECT_STATE_INTROSPECT_IN_PROGRESS
            self._pending_introspect = self._Introspect()
    bus_name = property(lambda self: self._named_service, None, None, 'The bus name to which this proxy is bound. (Read-only,\n            may change.)\n\n            If the proxy was instantiated using a unique name, this property\n            is that unique name.\n\n            If the proxy was instantiated with a well-known name and with\n            ``follow_name_owner_changes`` set false (the default), this\n            property is the unique name of the connection that owned that\n            well-known name when the proxy was instantiated, which might\n            not actually own the requested well-known name any more.\n\n            If the proxy was instantiated with a well-known name and with\n            ``follow_name_owner_changes`` set true, this property is that\n            well-known name.\n            ')
    requested_bus_name = property(lambda self: self._requested_bus_name, None, None, 'The bus name which was requested when this proxy was\n            instantiated.\n            ')
    object_path = property(lambda self: self.__dbus_object_path__, None, None, 'The object-path of this proxy.')

    def connect_to_signal(self, signal_name, handler_function, dbus_interface=None, **keywords):
        """Arrange for the given function to be called when the given signal
        is received.

        :Parameters:
            `signal_name` : str
                The name of the signal
            `handler_function` : callable
                A function to be called when the signal is emitted by
                the remote object. Its positional arguments will be the
                arguments of the signal; optionally, it may be given
                keyword arguments as described below.
            `dbus_interface` : str
                Optional interface with which to qualify the signal name.
                If None (the default) the handler will be called whenever a
                signal of the given member name is received, whatever
                its interface.
        :Keywords:
            `utf8_strings` : bool
                If True, the handler function will receive any string
                arguments as dbus.UTF8String objects (a subclass of str
                guaranteed to be UTF-8). If False (default) it will receive
                any string arguments as dbus.String objects (a subclass of
                unicode).
            `byte_arrays` : bool
                If True, the handler function will receive any byte-array
                arguments as dbus.ByteArray objects (a subclass of str).
                If False (default) it will receive any byte-array
                arguments as a dbus.Array of dbus.Byte (subclasses of:
                a list of ints).
            `sender_keyword` : str
                If not None (the default), the handler function will receive
                the unique name of the sending endpoint as a keyword
                argument with this name
            `destination_keyword` : str
                If not None (the default), the handler function will receive
                the bus name of the destination (or None if the signal is a
                broadcast, as is usual) as a keyword argument with this name.
            `interface_keyword` : str
                If not None (the default), the handler function will receive
                the signal interface as a keyword argument with this name.
            `member_keyword` : str
                If not None (the default), the handler function will receive
                the signal name as a keyword argument with this name.
            `path_keyword` : str
                If not None (the default), the handler function will receive
                the object-path of the sending object as a keyword argument
                with this name
            `message_keyword` : str
                If not None (the default), the handler function will receive
                the `dbus.lowlevel.SignalMessage` as a keyword argument with
                this name.
            `arg...` : unicode or UTF-8 str
                If there are additional keyword parameters of the form
                ``arg``\\ *n*, match only signals where the *n*\\ th argument
                is the value given for that keyword parameter. As of this time
                only string arguments can be matched (in particular,
                object paths and signatures can't).
        """
        return self._bus.add_signal_receiver(handler_function, signal_name=signal_name, dbus_interface=dbus_interface, bus_name=self._named_service, path=self.__dbus_object_path__, **keywords)

    def _Introspect(self):
        kwargs = {}
        return self._bus.call_async(self._named_service, self.__dbus_object_path__, INTROSPECTABLE_IFACE, 'Introspect', '', (), self._introspect_reply_handler, self._introspect_error_handler, require_main_loop=False, **kwargs)

    def _introspect_execute_queue(self):
        for proxy_method, args, keywords in self._pending_introspect_queue:
            proxy_method(*args, **keywords)
        self._pending_introspect_queue = []

    def _introspect_reply_handler(self, data):
        self._introspect_lock.acquire()
        try:
            try:
                self._introspect_method_map = process_introspection_data(data)
            except IntrospectionParserException as e:
                self._introspect_error_handler(e)
                return
            self._introspect_state = self.INTROSPECT_STATE_INTROSPECT_DONE
            self._pending_introspect = None
            self._introspect_execute_queue()
        finally:
            self._introspect_lock.release()

    def _introspect_error_handler(self, error):
        logging.basicConfig()
        _logger.error('Introspect error on %s:%s: %s.%s: %s', self._named_service, self.__dbus_object_path__, error.__class__.__module__, error.__class__.__name__, error)
        self._introspect_lock.acquire()
        try:
            _logger.debug('Executing introspect queue due to error')
            self._introspect_state = self.INTROSPECT_STATE_DONT_INTROSPECT
            self._pending_introspect = None
            self._introspect_execute_queue()
        finally:
            self._introspect_lock.release()

    def _introspect_block(self):
        self._introspect_lock.acquire()
        try:
            if self._pending_introspect is not None:
                self._pending_introspect.block()
        finally:
            self._introspect_lock.release()

    def _introspect_add_to_queue(self, callback, args, kwargs):
        self._introspect_lock.acquire()
        try:
            if self._introspect_state == self.INTROSPECT_STATE_INTROSPECT_IN_PROGRESS:
                self._pending_introspect_queue.append((callback, args, kwargs))
            else:
                callback(*args, **kwargs)
        finally:
            self._introspect_lock.release()

    def __getattr__(self, member):
        if member.startswith('__') and member.endswith('__'):
            raise AttributeError(member)
        else:
            return self.get_dbus_method(member)

    def get_dbus_method(self, member, dbus_interface=None):
        """Return a proxy method representing the given D-Bus method. The
        returned proxy method can be called in the usual way. For instance, ::

            proxy.get_dbus_method("Foo", dbus_interface='com.example.Bar')(123)

        is equivalent to::

            proxy.Foo(123, dbus_interface='com.example.Bar')

        or even::

            getattr(proxy, "Foo")(123, dbus_interface='com.example.Bar')

        However, using `get_dbus_method` is the only way to call D-Bus
        methods with certain awkward names - if the author of a service
        implements a method called ``connect_to_signal`` or even
        ``__getattr__``, you'll need to use `get_dbus_method` to call them.

        For services which follow the D-Bus convention of CamelCaseMethodNames
        this won't be a problem.
        """
        ret = self.ProxyMethodClass(self, self._bus, self._named_service, self.__dbus_object_path__, member, dbus_interface)
        if self._introspect_state == self.INTROSPECT_STATE_INTROSPECT_IN_PROGRESS:
            ret = self.DeferredMethodClass(ret, self._introspect_add_to_queue, self._introspect_block)
        return ret

    def __repr__(self):
        return '<ProxyObject wrapping %s %s %s at %#x>' % (self._bus, self._named_service, self.__dbus_object_path__, id(self))
    __str__ = __repr__