from sys import version_info as _swig_python_version_info
import weakref
class SequenceVar(PropagationBaseObject):
    """
    A sequence variable is a variable whose domain is a set of possible
    orderings of the interval variables. It allows ordering of tasks. It
    has two sets of methods: ComputePossibleFirstsAndLasts(), which
    returns the list of interval variables that can be ranked first or
    last; and RankFirst/RankNotFirst/RankLast/RankNotLast, which can be
    used to create the search decision.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')

    def DebugString(self):
        return _pywrapcp.SequenceVar_DebugString(self)

    def RankFirst(self, index):
        """
        Ranks the index_th interval var first of all unranked interval
        vars. After that, it will no longer be considered ranked.
        """
        return _pywrapcp.SequenceVar_RankFirst(self, index)

    def RankNotFirst(self, index):
        """
        Indicates that the index_th interval var will not be ranked first
        of all currently unranked interval vars.
        """
        return _pywrapcp.SequenceVar_RankNotFirst(self, index)

    def RankLast(self, index):
        """
        Ranks the index_th interval var first of all unranked interval
        vars. After that, it will no longer be considered ranked.
        """
        return _pywrapcp.SequenceVar_RankLast(self, index)

    def RankNotLast(self, index):
        """
        Indicates that the index_th interval var will not be ranked first
        of all currently unranked interval vars.
        """
        return _pywrapcp.SequenceVar_RankNotLast(self, index)

    def Interval(self, index):
        """ Returns the index_th interval of the sequence."""
        return _pywrapcp.SequenceVar_Interval(self, index)

    def Next(self, index):
        """ Returns the next of the index_th interval of the sequence."""
        return _pywrapcp.SequenceVar_Next(self, index)

    def Size(self):
        """ Returns the number of interval vars in the sequence."""
        return _pywrapcp.SequenceVar_Size(self)

    def __repr__(self):
        return _pywrapcp.SequenceVar___repr__(self)

    def __str__(self):
        return _pywrapcp.SequenceVar___str__(self)