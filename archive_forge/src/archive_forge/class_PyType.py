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
class PyType(PyObj):

    def __init__(self, name=None, pyname=None, root=None, superior=None, internal=True, namespace=None):
        PyObj.__init__(self, name, pyname, root)
        self.class_name = leading_uppercase(self.name + '_')
        self.properties = ([], [])
        if superior:
            self.superior = [superior]
        else:
            self.superior = []
        self.value_type = None
        self.internal = internal
        self.namespace = namespace

    def text(self, target_namespace, cdict, _child=True, ignore=None, _session=None):
        if not self.properties and (not self.type) and (not self.superior):
            self.done = True
            return ([], self.class_definition(target_namespace, cdict))
        if ignore is None:
            ignore = []
        req = []
        inherited_properties = []
        for sup in self.superior:
            try:
                supc = cdict[sup]
            except KeyError:
                mod, typ = sup.split('.')
                supc = pyobj_factory(sup, None, None)
                if mod:
                    supc.properties = [_import_attrs(self.root.modul[mod], typ, self.root), []]
                cdict[sup] = supc
                supc.done = True
            if not supc.done:
                res = _do(supc, target_namespace, cdict, req)
                if isinstance(res, tuple):
                    return res
            if not self.properties[1]:
                inherited_properties = reqursive_superior(supc, cdict)
        if inherited_properties:
            self.properties = (self.properties[0], rm_duplicates(inherited_properties))
        own, inh = self.properties
        own = rm_duplicates(expand_groups(own, cdict))
        self.properties = (own, inh)
        for prop in own:
            if not prop.name:
                continue
            if not prop.done:
                if prop.name in ignore:
                    continue
                res = _do(prop, target_namespace, cdict, req)
                if res == ([], None):
                    res = (req, None)
                if isinstance(res, tuple):
                    return res
        return (req, self.class_definition(target_namespace, cdict, ignore))

    def undefined(self, cdict):
        undef = ([], [])
        for sup in self.superior:
            supc = cdict[sup]
            if not supc.done:
                undef[0].append(supc)
        own, _ = self.properties
        for prop in own:
            if not prop.name:
                continue
            if isinstance(prop, PyAttribute):
                continue
            if not prop.done:
                undef[1].append(prop)
        return undef