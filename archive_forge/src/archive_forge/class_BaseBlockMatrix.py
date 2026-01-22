class BaseBlockMatrix(object):
    """Base class for block matrices"""

    def __init__(self):
        pass

    def tolil(self, copy=False):
        msg = 'tolil not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def todia(self, copy=False):
        msg = 'todia not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def tobsr(self, blocksize=None, copy=False):
        msg = 'tobsr not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def sum(self, axis=None, dtype=None, out=None):
        msg = 'sum not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def mean(self, axis=None, dtype=None, out=None):
        msg = 'mean not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def diagonal(self, k=0):
        msg = 'diagonal not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def nonzero(self):
        msg = 'nonzero not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def setdiag(self, values, k=0):
        msg = 'setdiag not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def transpose(self, *axes):
        msg = 'transpose not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)

    def tostring(self, order='C'):
        msg = 'tostring not implemented for {}'.format(self.__class__.__name__)
        raise NotImplementedError(msg)