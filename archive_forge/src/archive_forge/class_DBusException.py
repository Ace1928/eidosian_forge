from dbus._compat import is_py3
class DBusException(Exception):
    include_traceback = False
    'If True, tracebacks will be included in the exception message sent to\n    D-Bus clients.\n\n    Exceptions that are not DBusException subclasses always behave\n    as though this is True. Set this to True on DBusException subclasses\n    that represent a programming error, and leave it False on subclasses that\n    represent an expected failure condition (e.g. a network server not\n    responding).'

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        if name is not None or getattr(self, '_dbus_error_name', None) is None:
            self._dbus_error_name = name
        if kwargs:
            raise TypeError('DBusException does not take keyword arguments: %s' % ', '.join(kwargs.keys()))
        Exception.__init__(self, *args)

    def __unicode__(self):
        """Return a unicode error"""
        if len(self.args) > 1:
            s = unicode(self.args)
        else:
            s = ''.join(self.args)
        if self._dbus_error_name is not None:
            return '%s: %s' % (self._dbus_error_name, s)
        else:
            return s

    def __str__(self):
        """Return a str error"""
        s = Exception.__str__(self)
        if self._dbus_error_name is not None:
            return '%s: %s' % (self._dbus_error_name, s)
        else:
            return s

    def get_dbus_message(self):
        if len(self.args) > 1:
            s = str(self.args)
        else:
            s = ''.join(self.args)
        if isinstance(s, bytes):
            return s.decode('utf-8', 'replace')
        return s

    def get_dbus_name(self):
        return self._dbus_error_name