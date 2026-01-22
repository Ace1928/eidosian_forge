import errno
import getopt
import importlib
import re
import sys
import time
import types
from xml.etree import ElementTree as ElementTree
import saml2
from saml2 import SamlBase
class PyAny(PyObj):

    def __init__(self, name=None, pyname=None, _external=False, _namespace=''):
        PyObj.__init__(self, name, pyname)
        self.done = True