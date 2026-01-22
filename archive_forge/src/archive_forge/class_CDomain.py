import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class CDomain(Domain):
    """C language domain."""
    name = 'c'
    label = 'C'
    object_types = {'member': ObjType(_('member'), 'var', 'member', 'data', 'identifier'), 'var': ObjType(_('variable'), 'var', 'member', 'data', 'identifier'), 'function': ObjType(_('function'), 'func', 'identifier', 'type'), 'macro': ObjType(_('macro'), 'macro', 'identifier'), 'struct': ObjType(_('struct'), 'struct', 'identifier', 'type'), 'union': ObjType(_('union'), 'union', 'identifier', 'type'), 'enum': ObjType(_('enum'), 'enum', 'identifier', 'type'), 'enumerator': ObjType(_('enumerator'), 'enumerator', 'identifier'), 'type': ObjType(_('type'), 'identifier', 'type'), 'functionParam': ObjType(_('function parameter'), 'identifier', 'var', 'member', 'data')}
    directives = {'member': CMemberObject, 'var': CMemberObject, 'function': CFunctionObject, 'macro': CMacroObject, 'struct': CStructObject, 'union': CUnionObject, 'enum': CEnumObject, 'enumerator': CEnumeratorObject, 'type': CTypeObject, 'namespace': CNamespaceObject, 'namespace-push': CNamespacePushObject, 'namespace-pop': CNamespacePopObject, 'alias': CAliasObject}
    roles = {'member': CXRefRole(), 'data': CXRefRole(), 'var': CXRefRole(), 'func': CXRefRole(fix_parens=True), 'macro': CXRefRole(), 'struct': CXRefRole(), 'union': CXRefRole(), 'enum': CXRefRole(), 'enumerator': CXRefRole(), 'type': CXRefRole(), 'expr': CExprRole(asCode=True), 'texpr': CExprRole(asCode=False)}
    initial_data: Dict[str, Union[Symbol, Dict[str, Tuple[str, str, str]]]] = {'root_symbol': Symbol(None, None, None, None, None), 'objects': {}}

    def clear_doc(self, docname: str) -> None:
        if Symbol.debug_show_tree:
            print('clear_doc:', docname)
            print('\tbefore:')
            print(self.data['root_symbol'].dump(1))
            print('\tbefore end')
        rootSymbol = self.data['root_symbol']
        rootSymbol.clear_doc(docname)
        if Symbol.debug_show_tree:
            print('\tafter:')
            print(self.data['root_symbol'].dump(1))
            print('\tafter end')
            print('clear_doc end:', docname)

    def process_doc(self, env: BuildEnvironment, docname: str, document: nodes.document) -> None:
        if Symbol.debug_show_tree:
            print('process_doc:', docname)
            print(self.data['root_symbol'].dump(0))
            print('process_doc end:', docname)

    def process_field_xref(self, pnode: pending_xref) -> None:
        pnode.attributes.update(self.env.ref_context)

    def merge_domaindata(self, docnames: List[str], otherdata: Dict) -> None:
        if Symbol.debug_show_tree:
            print('merge_domaindata:')
            print('\tself:')
            print(self.data['root_symbol'].dump(1))
            print('\tself end')
            print('\tother:')
            print(otherdata['root_symbol'].dump(1))
            print('\tother end')
            print('merge_domaindata end')
        self.data['root_symbol'].merge_with(otherdata['root_symbol'], docnames, self.env)
        ourObjects = self.data['objects']
        for fullname, (fn, id_, objtype) in otherdata['objects'].items():
            if fn in docnames:
                if fullname not in ourObjects:
                    ourObjects[fullname] = (fn, id_, objtype)

    def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Tuple[Optional[Element], Optional[str]]:
        parser = DefinitionParser(target, location=node, config=env.config)
        try:
            name = parser.parse_xref_object()
        except DefinitionError as e:
            logger.warning('Unparseable C cross-reference: %r\n%s', target, e, location=node)
            return (None, None)
        parentKey: LookupKey = node.get('c:parent_key', None)
        rootSymbol = self.data['root_symbol']
        if parentKey:
            parentSymbol: Symbol = rootSymbol.direct_lookup(parentKey)
            if not parentSymbol:
                print('Target: ', target)
                print('ParentKey: ', parentKey)
                print(rootSymbol.dump(1))
            assert parentSymbol
        else:
            parentSymbol = rootSymbol
        s = parentSymbol.find_declaration(name, typ, matchSelf=True, recurseInAnon=True)
        if s is None or s.declaration is None:
            return (None, None)
        declaration = s.declaration
        displayName = name.get_display_string()
        docname = s.docname
        assert docname
        return (make_refnode(builder, fromdocname, docname, declaration.get_newest_id(), contnode, displayName), declaration.objectType)

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        return self._resolve_xref_inner(env, fromdocname, builder, typ, target, node, contnode)[0]

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        with logging.suppress_logging():
            retnode, objtype = self._resolve_xref_inner(env, fromdocname, builder, 'any', target, node, contnode)
        if retnode:
            return [('c:' + self.role_for_objtype(objtype), retnode)]
        return []

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        rootSymbol = self.data['root_symbol']
        for symbol in rootSymbol.get_all_symbols():
            if symbol.declaration is None:
                continue
            assert symbol.docname
            fullNestedName = symbol.get_full_nested_name()
            name = str(fullNestedName).lstrip('.')
            dispname = fullNestedName.get_display_string().lstrip('.')
            objectType = symbol.declaration.objectType
            docname = symbol.docname
            newestId = symbol.declaration.get_newest_id()
            yield (name, dispname, objectType, docname, newestId, 1)