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
def _external_class(self, mod, typ, cdict, child, target_namespace, ignore):
    cname = self.root.modul[mod].factory(typ).__class__.__name__
    imp_name = f'{mod}.{cname}'
    if imp_name not in cdict:
        impo = pyelement_factory(imp_name, None, None)
        impo.properties = [_import_attrs(self.root.modul[mod], typ, self.root), []]
        impo.class_name = imp_name
        cdict[imp_name] = impo
        impo.done = True
        if child:
            impo.local = True
    self.superior = [imp_name]
    text = self.class_definition(target_namespace, cdict, ignore=ignore)
    return text