import inspect
import re
import sys
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.domains.python import PyAttribute
from sphinx.domains.python import PyClasslike
from sphinx.domains.python import PyMethod
from sphinx.ext import autodoc
from sphinx.locale import _
from sphinx.roles import XRefRole
from sphinx.util.docfields import Field
from sphinx.util.nodes import make_refnode
import wsme
import wsme.rest.json
import wsme.rest.xml
import wsme.types
class TypeDirective(PyClasslike):

    def get_index_text(self, modname, name_cls):
        return _('%s (webservice type)') % name_cls[0]

    def add_target_and_index(self, name_cls, sig, signode):
        ret = super(TypeDirective, self).add_target_and_index(name_cls, sig, signode)
        name = name_cls[0]
        types = self.env.domaindata['wsme']['types']
        if name in types:
            self.state_machine.reporter.warning('duplicate type description of %s ' % name)
        types[name] = self.env.docname
        return ret