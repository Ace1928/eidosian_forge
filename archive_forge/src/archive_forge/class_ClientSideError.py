from wsme.utils import _
class ClientSideError(RuntimeError):

    def __init__(self, msg=None, status_code=400, faultcode='Client'):
        self.msg = msg
        self.code = status_code
        self.faultcode = faultcode
        super(ClientSideError, self).__init__(self.faultstring)

    @property
    def faultstring(self):
        if self.msg is None:
            return str(self)
        elif isinstance(self.msg, str):
            return self.msg
        else:
            return str(self.msg)