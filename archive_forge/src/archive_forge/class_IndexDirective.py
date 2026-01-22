from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.domains import Domain
from sphinx.environment import BuildEnvironment
from sphinx.util import logging, split_index_msg
from sphinx.util.docutils import ReferenceRole, SphinxDirective
from sphinx.util.nodes import process_index_entry
from sphinx.util.typing import OptionSpec
class IndexDirective(SphinxDirective):
    """
    Directive to add entries to the index.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec: OptionSpec = {'name': directives.unchanged}

    def run(self) -> List[Node]:
        arguments = self.arguments[0].split('\n')
        if 'name' in self.options:
            targetname = self.options['name']
            targetnode = nodes.target('', '', names=[targetname])
        else:
            targetid = 'index-%s' % self.env.new_serialno('index')
            targetnode = nodes.target('', '', ids=[targetid])
        self.state.document.note_explicit_target(targetnode)
        indexnode = addnodes.index()
        indexnode['entries'] = []
        indexnode['inline'] = False
        self.set_source_info(indexnode)
        for entry in arguments:
            indexnode['entries'].extend(process_index_entry(entry, targetnode['ids'][0]))
        return [indexnode, targetnode]