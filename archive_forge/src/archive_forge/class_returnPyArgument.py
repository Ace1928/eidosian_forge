import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class returnPyArgument(ReturnValues):
    """ReturnValues returning the named pyArgs value"""
    argNames = ('name',)
    indexLookups = [('index', 'name', 'pyArgIndex')]
    __slots__ = ('index', 'name')

    def __call__(self, result, baseOperation, pyArgs, cArgs):
        """Retrieve pyArgs[ self.index ]"""
        return pyArgs[self.index]