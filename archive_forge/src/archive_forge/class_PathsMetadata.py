from sys import version_info as _swig_python_version_info
import weakref
class PathsMetadata(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, manager):
        _pywrapcp.PathsMetadata_swiginit(self, _pywrapcp.new_PathsMetadata(manager))

    def IsStart(self, node):
        return _pywrapcp.PathsMetadata_IsStart(self, node)

    def IsEnd(self, node):
        return _pywrapcp.PathsMetadata_IsEnd(self, node)

    def GetPath(self, start_or_end_node):
        return _pywrapcp.PathsMetadata_GetPath(self, start_or_end_node)

    def NumPaths(self):
        return _pywrapcp.PathsMetadata_NumPaths(self)

    def Paths(self):
        return _pywrapcp.PathsMetadata_Paths(self)

    def Starts(self):
        return _pywrapcp.PathsMetadata_Starts(self)

    def Ends(self):
        return _pywrapcp.PathsMetadata_Ends(self)
    __swig_destroy__ = _pywrapcp.delete_PathsMetadata