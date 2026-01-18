import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_sourcecode_as_namespace():
    rcode = '\n'.join(('x <- 1+2', 'f <- function(x) x+1'))
    sc = robjects.packages.SourceCode(rcode)
    foo_ns = sc.as_namespace('foo_ns')
    assert hasattr(foo_ns, 'x')
    assert hasattr(foo_ns, 'f')
    assert foo_ns.x[0] == 3
    assert isinstance(foo_ns.f, rinterface.SexpClosure)