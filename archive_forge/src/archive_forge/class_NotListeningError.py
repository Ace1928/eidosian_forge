import socket
from incremental import Version
from twisted.python import deprecate
class NotListeningError(RuntimeError):
    __doc__ = MESSAGE = 'The Port was not listening when it was asked to stop listening'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s