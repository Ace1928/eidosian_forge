import re
from typing import (Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, TypeVar,
from docutils import nodes
from docutils.nodes import Element, Node, TextElement, system_message
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.errors import NoUri
from sphinx.locale import _, __
from sphinx.roles import SphinxRole, XRefRole
from sphinx.transforms import SphinxTransform
from sphinx.transforms.post_transforms import ReferencesResolver
from sphinx.util import logging
from sphinx.util.cfamily import (ASTAttributeList, ASTBaseBase, ASTBaseParenExprList,
from sphinx.util.docfields import Field, GroupedField
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import make_refnode
from sphinx.util.typing import OptionSpec
class CPPDomain(Domain):
    """C++ language domain.

    There are two 'object type' attributes being used::

    - Each object created from directives gets an assigned .objtype from ObjectDescription.run.
      This is simply the directive name.
    - Each declaration (see the distinction in the directives dict below) has a nested .ast of
      type ASTDeclaration. That object has .objectType which corresponds to the keys in the
      object_types dict below. They are the core different types of declarations in C++ that
      one can document.
    """
    name = 'cpp'
    label = 'C++'
    object_types = {'class': ObjType(_('class'), 'class', 'struct', 'identifier', 'type'), 'union': ObjType(_('union'), 'union', 'identifier', 'type'), 'function': ObjType(_('function'), 'func', 'identifier', 'type'), 'member': ObjType(_('member'), 'member', 'var', 'identifier'), 'type': ObjType(_('type'), 'identifier', 'type'), 'concept': ObjType(_('concept'), 'concept', 'identifier'), 'enum': ObjType(_('enum'), 'enum', 'identifier', 'type'), 'enumerator': ObjType(_('enumerator'), 'enumerator', 'identifier'), 'functionParam': ObjType(_('function parameter'), 'identifier', 'member', 'var'), 'templateParam': ObjType(_('template parameter'), 'identifier', 'class', 'struct', 'union', 'member', 'var', 'type')}
    directives = {'class': CPPClassObject, 'struct': CPPClassObject, 'union': CPPUnionObject, 'function': CPPFunctionObject, 'member': CPPMemberObject, 'var': CPPMemberObject, 'type': CPPTypeObject, 'concept': CPPConceptObject, 'enum': CPPEnumObject, 'enum-struct': CPPEnumObject, 'enum-class': CPPEnumObject, 'enumerator': CPPEnumeratorObject, 'namespace': CPPNamespaceObject, 'namespace-push': CPPNamespacePushObject, 'namespace-pop': CPPNamespacePopObject, 'alias': CPPAliasObject}
    roles = {'any': CPPXRefRole(), 'class': CPPXRefRole(), 'struct': CPPXRefRole(), 'union': CPPXRefRole(), 'func': CPPXRefRole(fix_parens=True), 'member': CPPXRefRole(), 'var': CPPXRefRole(), 'type': CPPXRefRole(), 'concept': CPPXRefRole(), 'enum': CPPXRefRole(), 'enumerator': CPPXRefRole(), 'expr': CPPExprRole(asCode=True), 'texpr': CPPExprRole(asCode=False)}
    initial_data = {'root_symbol': Symbol(None, None, None, None, None, None, None), 'names': {}}

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
        for name, nDocname in list(self.data['names'].items()):
            if nDocname == docname:
                del self.data['names'][name]

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
        self.data['root_symbol'].merge_with(otherdata['root_symbol'], docnames, self.env)
        ourNames = self.data['names']
        for name, docname in otherdata['names'].items():
            if docname in docnames:
                if name not in ourNames:
                    ourNames[name] = docname
        if Symbol.debug_show_tree:
            print('\tresult:')
            print(self.data['root_symbol'].dump(1))
            print('\tresult end')
            print('merge_domaindata end')

    def _resolve_xref_inner(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Tuple[Optional[Element], Optional[str]]:
        if typ in ('any', 'func'):
            target += '()'
        parser = DefinitionParser(target, location=node, config=env.config)
        try:
            ast, isShorthand = parser.parse_xref_object()
        except DefinitionError as e:

            def findWarning(e: Exception) -> Tuple[str, Exception]:
                if typ != 'any' and typ != 'func':
                    return (target, e)
                parser2 = DefinitionParser(target[:-2], location=node, config=env.config)
                try:
                    parser2.parse_xref_object()
                except DefinitionError as e2:
                    return (target[:-2], e2)
                return (target, e)
            t, ex = findWarning(e)
            logger.warning('Unparseable C++ cross-reference: %r\n%s', t, ex, location=node)
            return (None, None)
        parentKey: LookupKey = node.get('cpp:parent_key', None)
        rootSymbol = self.data['root_symbol']
        if parentKey:
            parentSymbol: Symbol = rootSymbol.direct_lookup(parentKey)
            if not parentSymbol:
                print('Target: ', target)
                print('ParentKey: ', parentKey.data)
                print(rootSymbol.dump(1))
            assert parentSymbol
        else:
            parentSymbol = rootSymbol
        if isShorthand:
            assert isinstance(ast, ASTNamespace)
            ns = ast
            name = ns.nestedName
            if ns.templatePrefix:
                templateDecls = ns.templatePrefix.templates
            else:
                templateDecls = []
            searchInSiblings = not name.rooted and len(name.names) == 1
            symbols, failReason = parentSymbol.find_name(name, templateDecls, typ, templateShorthand=True, matchSelf=True, recurseInAnon=True, searchInSiblings=searchInSiblings)
            if symbols is None:
                if typ == 'identifier':
                    if failReason == 'templateParamInQualified':
                        raise NoUri(str(name), typ)
                s = None
            else:
                s = symbols[0]
        else:
            assert isinstance(ast, ASTDeclaration)
            decl = ast
            name = decl.name
            s = parentSymbol.find_declaration(decl, typ, templateShorthand=True, matchSelf=True, recurseInAnon=True)
        if s is None or s.declaration is None:
            txtName = str(name)
            if txtName.startswith('std::') or txtName == 'std':
                raise NoUri(txtName, typ)
            return (None, None)
        if typ.startswith('cpp:'):
            typ = typ[4:]
        declTyp = s.declaration.objectType

        def checkType() -> bool:
            if typ == 'any':
                return True
            objtypes = self.objtypes_for_role(typ)
            if objtypes:
                return declTyp in objtypes
            print('Type is %s, declaration type is %s' % (typ, declTyp))
            raise AssertionError()
        if not checkType():
            logger.warning('cpp:%s targets a %s (%s).', typ, s.declaration.objectType, s.get_full_nested_name(), location=node)
        declaration = s.declaration
        if isShorthand:
            fullNestedName = s.get_full_nested_name()
            displayName = fullNestedName.get_display_string().lstrip(':')
        else:
            displayName = decl.get_display_string()
        docname = s.docname
        assert docname
        if typ != 'identifier':
            title = contnode.pop(0).astext()
            addParen = 0
            if not node.get('refexplicit', False) and declaration.objectType == 'function':
                if isShorthand:
                    if env.config.add_function_parentheses and typ == 'any':
                        addParen += 1
                    if env.config.add_function_parentheses and typ == 'func' and title.endswith('operator()'):
                        addParen += 1
                    if typ in ('any', 'func') and title.endswith('operator') and displayName.endswith('operator()'):
                        addParen += 1
                elif env.config.add_function_parentheses:
                    if typ == 'any' and displayName.endswith('()'):
                        addParen += 1
                    elif typ == 'func':
                        if title.endswith('()') and (not displayName.endswith('()')):
                            title = title[:-2]
                elif displayName.endswith('()'):
                    addParen += 1
            if addParen > 0:
                title += '()' * addParen
            contnode += nodes.Text(title)
        res = (make_refnode(builder, fromdocname, docname, declaration.get_newest_id(), contnode, displayName), declaration.objectType)
        return res

    def resolve_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, typ: str, target: str, node: pending_xref, contnode: Element) -> Optional[Element]:
        return self._resolve_xref_inner(env, fromdocname, builder, typ, target, node, contnode)[0]

    def resolve_any_xref(self, env: BuildEnvironment, fromdocname: str, builder: Builder, target: str, node: pending_xref, contnode: Element) -> List[Tuple[str, Element]]:
        with logging.suppress_logging():
            retnode, objtype = self._resolve_xref_inner(env, fromdocname, builder, 'any', target, node, contnode)
        if retnode:
            if objtype == 'templateParam':
                return [('cpp:templateParam', retnode)]
            else:
                return [('cpp:' + self.role_for_objtype(objtype), retnode)]
        return []

    def get_objects(self) -> Iterator[Tuple[str, str, str, str, str, int]]:
        rootSymbol = self.data['root_symbol']
        for symbol in rootSymbol.get_all_symbols():
            if symbol.declaration is None:
                continue
            assert symbol.docname
            fullNestedName = symbol.get_full_nested_name()
            name = str(fullNestedName).lstrip(':')
            dispname = fullNestedName.get_display_string().lstrip(':')
            objectType = symbol.declaration.objectType
            docname = symbol.docname
            newestId = symbol.declaration.get_newest_id()
            yield (name, dispname, objectType, docname, newestId, 1)

    def get_full_qualified_name(self, node: Element) -> str:
        target = node.get('reftarget', None)
        if target is None:
            return None
        parentKey: LookupKey = node.get('cpp:parent_key', None)
        if parentKey is None or len(parentKey.data) <= 0:
            return None
        rootSymbol = self.data['root_symbol']
        parentSymbol = rootSymbol.direct_lookup(parentKey)
        parentName = parentSymbol.get_full_nested_name()
        return '::'.join([str(parentName), target])