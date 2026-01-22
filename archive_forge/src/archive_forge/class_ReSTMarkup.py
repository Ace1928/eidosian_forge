import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils.nodes import Element
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.nodes import make_id, make_refnode
from sphinx.util.typing import OptionSpec
class ReSTMarkup(ObjectDescription[str]):
    """
    Description of generic reST markup.
    """
    option_spec: OptionSpec = {'noindex': directives.flag, 'noindexentry': directives.flag, 'nocontentsentry': directives.flag}

    def add_target_and_index(self, name: str, sig: str, signode: desc_signature) -> None:
        node_id = make_id(self.env, self.state.document, self.objtype, name)
        signode['ids'].append(node_id)
        self.state.document.note_explicit_target(signode)
        domain = cast(ReSTDomain, self.env.get_domain('rst'))
        domain.note_object(self.objtype, name, node_id, location=signode)
        if 'noindexentry' not in self.options:
            indextext = self.get_index_text(self.objtype, name)
            if indextext:
                self.indexnode['entries'].append(('single', indextext, node_id, '', None))

    def get_index_text(self, objectname: str, name: str) -> str:
        return ''

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for reST markups.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return self.objtype + '-' + name

    def _object_hierarchy_parts(self, sig_node: desc_signature) -> Tuple[str, ...]:
        if 'fullname' not in sig_node:
            return ()
        directive_names = []
        for parent in self.env.ref_context.get('rst:directives', ()):
            directive_names += parent.split(':')
        name = sig_node['fullname']
        return tuple(directive_names + name.split(':'))

    def _toc_entry_name(self, sig_node: desc_signature) -> str:
        if not sig_node.get('_toc_parts'):
            return ''
        config = self.env.app.config
        objtype = sig_node.parent.get('objtype')
        *parents, name = sig_node['_toc_parts']
        if objtype == 'directive:option':
            return f':{name}:'
        if config.toc_object_entries_show_parents in {'domain', 'all'}:
            name = ':'.join(sig_node['_toc_parts'])
        if objtype == 'role':
            return f':{name}:'
        if objtype == 'directive':
            return f'.. {name}::'
        return ''