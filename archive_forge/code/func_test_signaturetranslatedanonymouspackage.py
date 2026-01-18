import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_signaturetranslatedanonymouspackage():
    rcode = '\n    square <- function(x) {\n    return(x^2)\n    }\n    \n    cube <- function(x) {\n    return(x^3)\n    }\n    '
    powerpack = packages.STAP(rcode, 'powerpack')
    assert hasattr(powerpack, 'square')
    assert hasattr(powerpack, 'cube')