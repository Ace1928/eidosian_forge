from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
import oslo_i18n
from sphinx import addnodes
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain
from sphinx.domains import ObjType
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_refnode
from sphinx.util.nodes import nested_parse_with_titles
from oslo_config import cfg
from oslo_config import generator
class ConfigDomain(Domain):
    """oslo.config domain."""
    name = 'oslo.config'
    label = 'oslo.config'
    object_types = {'configoption': ObjType('configuration option', 'option')}
    directives = {'group': ConfigGroup, 'option': ConfigOption}
    roles = {'option': ConfigOptXRefRole(), 'group': ConfigGroupXRefRole()}
    initial_data = {'options': {}, 'groups': {}}

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        """Resolve cross-references"""
        if typ == 'option':
            group_name, option_name = target.split('.', 1)
            return make_refnode(builder, fromdocname, env.domaindata['oslo.config']['options'][target], target, contnode, option_name)
        if typ == 'group':
            return make_refnode(builder, fromdocname, env.domaindata['oslo.config']['groups'][target], target, contnode, target)
        return None

    def merge_domaindata(self, docnames, otherdata):
        for target, docname in otherdata['options'].items():
            if docname in docnames:
                self.data['options'][target] = docname
        for target, docname in otherdata['groups'].items():
            if docname in docnames:
                self.data['groups'][target] = docname