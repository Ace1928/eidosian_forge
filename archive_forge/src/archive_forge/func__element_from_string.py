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
def _element_from_string(self):
    print('ELEMENT_FROM_STRING = {')
    for elem in self.elems:
        if isinstance(elem, PyAttribute) or isinstance(elem, PyGroup):
            continue
        if elem.abstract:
            continue
        print(f'{INDENT}{elem.class_name}.c_tag: {pyify(elem.class_name)}_from_string,')
    print('}')
    print()