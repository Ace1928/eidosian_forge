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
class PyGroup:

    def __init__(self, name, root):
        self.name = name
        self.root = root
        self.properties = []
        self.done = False
        self.ref = []

    def text(self, _target_namespace, _dict, _child, _ignore):
        return ([], [])

    def undefined(self, _cdict):
        undef = ([], [])
        own, _ = self.properties
        for prop in own:
            if not prop.name:
                continue
            if not prop.done:
                undef[1].append(prop)
        return undef