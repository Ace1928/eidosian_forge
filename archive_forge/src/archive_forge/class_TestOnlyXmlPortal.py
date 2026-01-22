from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestOnlyXmlPortal(TestOnlyTextPortal):
    suffix = '.xml'
    skiplist = ['tableD', 'tableT', 'tableU']

    def create_options(self, name):
        return {'filename': os.path.abspath(tutorial_dir + os.sep + 'xml' + os.sep + name + self.suffix)}