import ctypes,logging
from OpenGL._bytes import bytes, unicode, as_8_bit
from OpenGL._null import NULL
from OpenGL import acceleratesupport
class returnCArgument(ReturnValues):
    """ReturnValues returning the named cArgs value"""
    argNames = ('name',)
    indexLookups = [('index', 'name', 'cArgIndex')]
    __slots__ = ('index', 'name')

    def __call__(self, result, baseOperation, pyArgs, cArgs):
        """Retrieve cArgs[ self.index ]"""
        return cArgs[self.index]