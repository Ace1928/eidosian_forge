import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def test_parsedcode():
    rcode = '1+2'
    expression = rinterface.parse(rcode)
    pc = robjects.packages.ParsedCode(expression)
    assert isinstance(pc, type(expression))