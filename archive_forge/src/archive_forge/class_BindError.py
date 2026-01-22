import socket
from incremental import Version
from twisted.python import deprecate
class BindError(Exception):
    __doc__ = MESSAGE = 'An error occurred binding to an interface'

    def __str__(self) -> str:
        s = self.MESSAGE
        if self.args:
            s = '{}: {}'.format(s, ' '.join(self.args))
        s = '%s.' % s
        return s