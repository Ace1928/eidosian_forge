import pytest
import sys
import textwrap
import rpy2.robjects as robjects
import rpy2.robjects.methods as methods
def test_set_accessors():
    robjects.r['setClass']('A', robjects.r('list(foo="numeric")'))
    robjects.r['setMethod']('length', signature='A', definition=robjects.r('function(x) 123'))

    class A(methods.RS4):

        def __init__(self):
            obj = robjects.r['new']('A')
            super().__init__(obj)
    acs = (('length', None, True, None),)
    methods.set_accessors(A, 'A', None, acs)
    a = A()
    assert a.length[0] == 123