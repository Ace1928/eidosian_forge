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
class WSMEDomain(Domain):
    name = 'wsme'
    label = 'WSME'
    object_types = {'type': ObjType(_('type'), 'type', 'obj'), 'service': ObjType(_('service'), 'service', 'obj')}
    directives = {'type': TypeDirective, 'attribute': AttributeDirective, 'service': ServiceDirective, 'root': RootDirective, 'function': FunctionDirective}
    roles = {'type': XRefRole()}
    initial_data = {'types': {}}

    def clear_doc(self, docname):
        keys = list(self.data['types'].keys())
        for key in keys:
            value = self.data['types'][key]
            if value == docname:
                del self.data['types'][key]

    def resolve_xref(self, env, fromdocname, builder, type, target, node, contnode):
        if target not in self.data['types']:
            return None
        todocname = self.data['types'][target]
        return make_refnode(builder, fromdocname, todocname, target, contnode, target)