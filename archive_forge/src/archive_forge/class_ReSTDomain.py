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
class ReSTDomain(Domain):
    """ReStructuredText domain."""
    name = 'rst'
    label = 'reStructuredText'
    object_types = {'directive': ObjType(_('directive'), 'dir'), 'directive:option': ObjType(_('directive-option'), 'dir'), 'role': ObjType(_('role'), 'role')}
    directives = {'directive': ReSTDirective, 'directive:option': ReSTDirectiveOption, 'role': ReSTRole}
    roles = {'dir': XRefRole(), 'role': XRefRole()}
    initial_data: Dict[str, Dict[Tuple[str, str], str]] = {'objects': {}}

    @property
    def objects(self) -> Dict[Tuple[str, str], Tuple[str, str]]:
        return self.data.setdefault('objects', {})

    def note_object(self, objtype: str, name: str, node_id: str, location: Any=None) -> None:
        if (objtype, name) in self.objects:
            docname, node_id = self.objects[objtype, name]
            logger.warning(__('duplicate description of %s %s, other instance in %s') % (objtype, name, docname), location=location)
        self.objects[objtype, name] = (self.env.docname, node_id)

    def clear_doc(self, docname: str) -> None:
        for (typ, name), (doc, _node_id) in list(self.objects.items()):
            if doc == docname:
                del self.objects[typ, name]

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for (typ, name), (doc, node_id) in otherdata['objects'].items():
            if doc in docnames:
                self.objects[typ, name] = (doc, node_id)

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        objtypes = self.objtypes_for_role(typ)
        for objtype in objtypes:
            todocname, node_id = self.objects.get((objtype, target), (None, None))
            if todocname:
                return make_refnode(builder, fromdocname, todocname, node_id, contnode, target + ' ' + objtype)
        return None

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        results: List[Tuple[str, Element]] = []
        for objtype in self.object_types:
            todocname, node_id = self.objects.get((objtype, target), (None, None))
            if todocname:
                results.append(('rst:' + self.role_for_objtype(objtype), make_refnode(builder, fromdocname, todocname, node_id, contnode, target + ' ' + objtype)))
        return results

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        for (typ, name), (docname, node_id) in self.data['objects'].items():
            yield (name, name, typ, docname, node_id, 1)