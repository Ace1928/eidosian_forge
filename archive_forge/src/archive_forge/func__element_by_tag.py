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
def _element_by_tag(self):
    print('ELEMENT_BY_TAG = {')
    listed = []
    for elem in self.elems:
        if isinstance(elem, PyAttribute) or isinstance(elem, PyGroup):
            continue
        if elem.abstract:
            continue
        lcen = elem.name
        print(f"{INDENT}'{lcen}': {elem.class_name},")
        listed.append(lcen)
    for elem in self.elems:
        if isinstance(elem, PyAttribute) or isinstance(elem, PyGroup):
            continue
        lcen = elem.name
        if elem.abstract and lcen not in listed:
            print(f"{INDENT}'{lcen}': {elem.class_name},")
            listed.append(lcen)
    print('}')
    print