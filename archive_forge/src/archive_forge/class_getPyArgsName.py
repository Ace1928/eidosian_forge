import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class getPyArgsName(CConverter):
    """CConverter returning named Python argument

        Intended for use in cConverters, the function returned
        retrieves the named pyArg and returns it when called.
        """
    argNames = ('name',)
    indexLookups = [('index', 'name', 'pyArgIndex')]
    __slots__ = ('index', 'name')

    def __call__(self, pyArgs, index, baseOperation):
        """Return pyArgs[ self.index ]"""
        try:
            return pyArgs[self.index]
        except AttributeError:
            raise RuntimeError('"Did not resolve parameter index for %r' % self.name)