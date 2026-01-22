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
def _import_attrs(modul, tag, top):
    obj = modul.factory(tag)
    properties = [PyAttribute(key, val[0], top, True, obj.c_namespace, val[2], val[1]) for key, val in obj.c_attributes.items()]
    for child in obj.c_child_order:
        for key, val in obj.c_children.items():
            pyn, mul = val
            maximum = 1
            if isinstance(mul, list):
                mul = mul[0]
                maximum = 'unbounded'
            if pyn == child:
                cpy = PyElement(name=mul.c_tag, pyname=pyn, root=top)
                cpy.max = maximum
                properties.append(cpy)
    return properties