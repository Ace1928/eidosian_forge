from sys import version_info as _swig_python_version_info
import weakref
class SearchLimit(SearchMonitor):
    """ Base class of all search limits."""
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr
    __swig_destroy__ = _pywrapcp.delete_SearchLimit

    def Crossed(self):
        """ Returns true if the limit has been crossed."""
        return _pywrapcp.SearchLimit_Crossed(self)

    def Check(self):
        """
        This method is called to check the status of the limit. A return
        value of true indicates that we have indeed crossed the limit. In
        that case, this method will not be called again and the remaining
        search will be discarded.
        """
        return _pywrapcp.SearchLimit_Check(self)

    def Init(self):
        """ This method is called when the search limit is initialized."""
        return _pywrapcp.SearchLimit_Init(self)

    def EnterSearch(self):
        """ Internal methods."""
        return _pywrapcp.SearchLimit_EnterSearch(self)

    def BeginNextDecision(self, b):
        return _pywrapcp.SearchLimit_BeginNextDecision(self, b)

    def RefuteDecision(self, d):
        return _pywrapcp.SearchLimit_RefuteDecision(self, d)

    def DebugString(self):
        return _pywrapcp.SearchLimit_DebugString(self)