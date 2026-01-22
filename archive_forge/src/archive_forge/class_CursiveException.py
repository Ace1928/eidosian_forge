from cursive.i18n import _
class CursiveException(Exception):
    """Base Cursive Exception

    To correctly use this class, inherit from it and define
    a 'msg_fmt' property. That msg_fmt will get printf'd
    with the keyword arguments provided to the constructor.

    """
    msg_fmt = _('An unknown exception occurred.')
    headers = {}
    safe = False

    def __init__(self, message=None, **kwargs):
        self.kwargs = kwargs
        if not message:
            try:
                message = self.msg_fmt % kwargs
            except Exception:
                message = self.msg_fmt
        self.message = message
        super(CursiveException, self).__init__(message)

    def format_message(self):
        return self.args[0]