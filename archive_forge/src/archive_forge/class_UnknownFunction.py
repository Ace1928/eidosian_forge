from wsme.utils import _
class UnknownFunction(ClientSideError):

    def __init__(self, name):
        self.name = name
        super(UnknownFunction, self).__init__()

    @property
    def faultstring(self):
        return _('Unknown function name: %s') % self.name