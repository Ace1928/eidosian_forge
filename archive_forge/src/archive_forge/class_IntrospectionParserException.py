from dbus._compat import is_py3
class IntrospectionParserException(DBusException):
    include_traceback = True

    def __init__(self, msg=''):
        DBusException.__init__(self, 'Error parsing introspect data: %s' % msg)