import os
import unittest
from distutils.core import PyPIRCCommand
from distutils.core import Distribution
from distutils.log import set_threshold
from distutils.log import WARN
from distutils.tests import support
class command(PyPIRCCommand):

    def __init__(self, dist):
        PyPIRCCommand.__init__(self, dist)

    def initialize_options(self):
        pass
    finalize_options = initialize_options