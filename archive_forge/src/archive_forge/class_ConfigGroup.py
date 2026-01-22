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
class ConfigGroup(rst.Directive):
    required_arguments = 1
    optional_arguments = 0
    has_content = True
    option_spec = {'namespace': directives.unchanged}

    def run(self):
        env = self.state.document.settings.env
        group_name = self.arguments[0]
        namespace = self.options.get('namespace')
        cached_groups = env.domaindata['oslo.config']['groups']
        env.temp_data['oslo.config:group'] = group_name
        LOG.debug('oslo.config group %s' % group_name)
        cached_groups[group_name] = env.docname
        result = ViewList()
        source_name = '<' + __name__ + '>'

        def _add(text):
            """Append some text to the output result view to be parsed."""
            result.append(text, source_name)
        if namespace:
            title = '%s: %s' % (namespace, group_name)
        else:
            title = group_name
        _add(title)
        _add('-' * len(title))
        _add('')
        for line in self.content:
            _add(line)
        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, result, node)
        first_child = node.children[0]
        target_name = cfg._normalize_group_name(group_name)
        first_child['ids'].append(target_name)
        indexnode = addnodes.index(entries=[])
        return [indexnode] + node.children