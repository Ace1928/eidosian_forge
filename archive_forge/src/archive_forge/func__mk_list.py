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
def _mk_list(self, objekt, alla, eldict):
    tup = []
    for prop in alla:
        mod, cname = _mod_cname(prop, eldict)
        if prop.max == 'unbounded':
            lista = True
        else:
            lista = False
        spec = objekt.child_spec(self.target_namespace, prop, mod, cname, lista)
        lines = [f'{objekt.class_name}.{spec}']
        tup.append((prop, lines, spec))
    return tup