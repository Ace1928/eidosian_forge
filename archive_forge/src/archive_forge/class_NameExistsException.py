from dbus._compat import is_py3
class NameExistsException(DBusException):
    include_traceback = True

    def __init__(self, name):
        DBusException.__init__(self, 'Bus name already exists: %s' % name)