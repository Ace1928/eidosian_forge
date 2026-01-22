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
class ReSTDirectiveOption(ReSTMarkup):
    """
    Description of an option for reST directive.
    """
    option_spec: OptionSpec = ReSTMarkup.option_spec.copy()
    option_spec.update({'type': directives.unchanged})

    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        try:
            name, argument = re.split('\\s*:\\s+', sig.strip(), 1)
        except ValueError:
            name, argument = (sig, None)
        desc_name = f':{name}:'
        signode['fullname'] = name.strip()
        signode += addnodes.desc_name(desc_name, desc_name)
        if argument:
            signode += addnodes.desc_annotation(' ' + argument, ' ' + argument)
        if self.options.get('type'):
            text = ' (%s)' % self.options['type']
            signode += addnodes.desc_annotation(text, text)
        return name

    def add_target_and_index(self, name: str, sig: str, signode: desc_signature) -> None:
        domain = cast(ReSTDomain, self.env.get_domain('rst'))
        directive_name = self.current_directive
        if directive_name:
            prefix = '-'.join([self.objtype, directive_name])
            objname = ':'.join([directive_name, name])
        else:
            prefix = self.objtype
            objname = name
        node_id = make_id(self.env, self.state.document, prefix, name)
        signode['ids'].append(node_id)
        self.state.document.note_explicit_target(signode)
        domain.note_object(self.objtype, objname, node_id, location=signode)
        if directive_name:
            key = name[0].upper()
            pair = [_('%s (directive)') % directive_name, _(':%s: (directive option)') % name]
            self.indexnode['entries'].append(('pair', '; '.join(pair), node_id, '', key))
        else:
            key = name[0].upper()
            text = _(':%s: (directive option)') % name
            self.indexnode['entries'].append(('single', text, node_id, '', key))

    @property
    def current_directive(self) -> str:
        directives = self.env.ref_context.get('rst:directives')
        if directives:
            return directives[-1]
        else:
            return ''

    def make_old_id(self, name: str) -> str:
        """Generate old styled node_id for directive options.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return '-'.join([self.objtype, self.current_directive, name])