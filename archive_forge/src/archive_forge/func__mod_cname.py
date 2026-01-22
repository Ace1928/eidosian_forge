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
def _mod_cname(prop, cdict):
    if hasattr(prop, 'scoped'):
        cname = prop.class_name
        mod = None
    else:
        mod, typ = _mod_typ(prop)
        if not mod:
            try:
                cname = cdict[typ].class_name
            except KeyError:
                cname = cdict[class_pyify(typ)].class_name
        else:
            cname = typ
    return (mod, cname)