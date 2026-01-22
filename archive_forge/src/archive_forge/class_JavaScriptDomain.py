from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.domains.python import _pseudo_parse_arglist
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.nodes import make_id, make_refnode, nested_parse_with_titles
from sphinx.util.typing import OptionSpec
class JavaScriptDomain(Domain):
    """JavaScript language domain."""
    name = 'js'
    label = 'JavaScript'
    object_types = {'function': ObjType(_('function'), 'func'), 'method': ObjType(_('method'), 'meth'), 'class': ObjType(_('class'), 'class'), 'data': ObjType(_('data'), 'data'), 'attribute': ObjType(_('attribute'), 'attr'), 'module': ObjType(_('module'), 'mod')}
    directives = {'function': JSCallable, 'method': JSCallable, 'class': JSConstructor, 'data': JSObject, 'attribute': JSObject, 'module': JSModule}
    roles = {'func': JSXRefRole(fix_parens=True), 'meth': JSXRefRole(fix_parens=True), 'class': JSXRefRole(fix_parens=True), 'data': JSXRefRole(), 'attr': JSXRefRole(), 'mod': JSXRefRole()}
    initial_data: Dict[str, Dict[str, Tuple[str, str]]] = {'objects': {}, 'modules': {}}

    @property
    def objects(self) -> Dict[str, Tuple[str, str, str]]:
        return self.data.setdefault('objects', {})

    def note_object(self, fullname: str, objtype: str, node_id: str, location: Any=None) -> None:
        if fullname in self.objects:
            docname = self.objects[fullname][0]
            logger.warning(__('duplicate %s description of %s, other %s in %s'), objtype, fullname, objtype, docname, location=location)
        self.objects[fullname] = (self.env.docname, node_id, objtype)

    @property
    def modules(self) -> Dict[str, Tuple[str, str]]:
        return self.data.setdefault('modules', {})

    def note_module(self, modname: str, node_id: str) -> None:
        self.modules[modname] = (self.env.docname, node_id)

    def clear_doc(self, docname: str) -> None:
        for fullname, (pkg_docname, _node_id, _l) in list(self.objects.items()):
            if pkg_docname == docname:
                del self.objects[fullname]
        for modname, (pkg_docname, _node_id) in list(self.modules.items()):
            if pkg_docname == docname:
                del self.modules[modname]

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        for fullname, (fn, node_id, objtype) in otherdata['objects'].items():
            if fn in docnames:
                self.objects[fullname] = (fn, node_id, objtype)
        for mod_name, (pkg_docname, node_id) in otherdata['modules'].items():
            if pkg_docname in docnames:
                self.modules[mod_name] = (pkg_docname, node_id)

    def find_obj(self, env: BuildEnvironment, mod_name: str, prefix: str, name: str, typ: str, searchorder: int=0) -> Tuple[str, Tuple[str, str, str]]:
        if name[-2:] == '()':
            name = name[:-2]
        searches = []
        if mod_name and prefix:
            searches.append('.'.join([mod_name, prefix, name]))
        if mod_name:
            searches.append('.'.join([mod_name, name]))
        if prefix:
            searches.append('.'.join([prefix, name]))
        searches.append(name)
        if searchorder == 0:
            searches.reverse()
        newname = None
        for search_name in searches:
            if search_name in self.objects:
                newname = search_name
        return (newname, self.objects.get(newname))

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        mod_name = node.get('js:module')
        prefix = node.get('js:object')
        searchorder = 1 if node.hasattr('refspecific') else 0
        name, obj = self.find_obj(env, mod_name, prefix, target, typ, searchorder)
        if not obj:
            return None
        return make_refnode(builder, fromdocname, obj[0], obj[1], contnode, name)

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        mod_name = node.get('js:module')
        prefix = node.get('js:object')
        name, obj = self.find_obj(env, mod_name, prefix, target, None, 1)
        if not obj:
            return []
        return [('js:' + self.role_for_objtype(obj[2]), make_refnode(builder, fromdocname, obj[0], obj[1], contnode, name))]

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        for refname, (docname, node_id, typ) in list(self.objects.items()):
            yield (refname, refname, typ, docname, node_id, 1)

    def get_full_qualified_name(self, node: Element) -> str:
        modname = node.get('js:module')
        prefix = node.get('js:object')
        target = node.get('reftarget')
        if target is None:
            return None
        else:
            return '.'.join(filter(None, [modname, prefix, target]))