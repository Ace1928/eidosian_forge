from zope.interface import Interface
class IReadWriteHandle(IReadHandle, IWriteHandle):
    pass