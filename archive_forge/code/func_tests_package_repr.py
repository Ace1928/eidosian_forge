import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
def tests_package_repr(self):
    env = robjects.Environment()
    pck = robjects.packages.Package(env, 'dummy_package')
    assert isinstance(repr(pck), str)