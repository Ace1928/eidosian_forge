from io import BytesIO
from .. import tests
from ..bzr.smart import medium, protocol
from ..transport import chroot, memory
from ..transport.http import wsgi
class FakeTransport:

    def __init__(self):
        self.calls = []
        self.base = 'fake:///'

    def abspath(self, relpath):
        return 'fake:///' + relpath

    def clone(self, relpath):
        self.calls.append(('clone', relpath))
        return self