from sys import version_info as _swig_python_version_info
import weakref
class SequenceVarElement(AssignmentElement):
    """
    The SequenceVarElement stores a partial representation of ranked
    interval variables in the underlying sequence variable.
    This representation consists of three vectors:
      - the forward sequence. That is the list of interval variables
        ranked first in the sequence.  The first element of the backward
        sequence is the first interval in the sequence variable.
      - the backward sequence. That is the list of interval variables
        ranked last in the sequence. The first element of the backward
        sequence is the last interval in the sequence variable.
      - The list of unperformed interval variables.
     Furthermore, if all performed variables are ranked, then by
     convention, the forward_sequence will contain all such variables
     and the backward_sequence will be empty.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined')
    __repr__ = _swig_repr

    def Var(self):
        return _pywrapcp.SequenceVarElement_Var(self)

    def ForwardSequence(self):
        return _pywrapcp.SequenceVarElement_ForwardSequence(self)

    def BackwardSequence(self):
        return _pywrapcp.SequenceVarElement_BackwardSequence(self)

    def Unperformed(self):
        return _pywrapcp.SequenceVarElement_Unperformed(self)

    def SetSequence(self, forward_sequence, backward_sequence, unperformed):
        return _pywrapcp.SequenceVarElement_SetSequence(self, forward_sequence, backward_sequence, unperformed)

    def SetForwardSequence(self, forward_sequence):
        return _pywrapcp.SequenceVarElement_SetForwardSequence(self, forward_sequence)

    def SetBackwardSequence(self, backward_sequence):
        return _pywrapcp.SequenceVarElement_SetBackwardSequence(self, backward_sequence)

    def SetUnperformed(self, unperformed):
        return _pywrapcp.SequenceVarElement_SetUnperformed(self, unperformed)

    def __eq__(self, element):
        return _pywrapcp.SequenceVarElement___eq__(self, element)

    def __ne__(self, element):
        return _pywrapcp.SequenceVarElement___ne__(self, element)
    __swig_destroy__ = _pywrapcp.delete_SequenceVarElement