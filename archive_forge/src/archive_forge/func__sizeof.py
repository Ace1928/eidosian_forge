from __future__ import print_function
import sys
import traceback
import pdb
import inspect
from .core import Construct, Subconstruct
from .lib import HexString, Container, ListContainer
def _sizeof(self, context):
    return 0