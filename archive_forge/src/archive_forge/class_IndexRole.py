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
class IndexRole(ReferenceRole):

    def run(self) -> Tuple[List[Node], List[system_message]]:
        target_id = 'index-%s' % self.env.new_serialno('index')
        if self.has_explicit_title:
            title = self.title
            entries = process_index_entry(self.target, target_id)
        elif self.target.startswith('!'):
            title = self.title[1:]
            entries = [('single', self.target[1:], target_id, 'main', None)]
        else:
            title = self.title
            entries = [('single', self.target, target_id, '', None)]
        index = addnodes.index(entries=entries)
        target = nodes.target('', '', ids=[target_id])
        text = nodes.Text(title)
        self.set_source_info(index)
        return ([index, target, text], [])