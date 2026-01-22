from dbus._compat import is_py3
class MissingErrorHandlerException(DBusException):
    include_traceback = True

    def __init__(self):
        DBusException.__init__(self, 'error_handler not defined: if you define a reply_handler you must also define an error_handler')