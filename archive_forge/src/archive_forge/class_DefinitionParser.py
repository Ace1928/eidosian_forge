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
class DefinitionParser(BaseParser):

    @property
    def language(self) -> str:
        return 'C'

    @property
    def id_attributes(self):
        return self.config.c_id_attributes

    @property
    def paren_attributes(self):
        return self.config.c_paren_attributes

    def _parse_string(self) -> str:
        if self.current_char != '"':
            return None
        startPos = self.pos
        self.pos += 1
        escape = False
        while True:
            if self.eof:
                self.fail('Unexpected end during inside string.')
            elif self.current_char == '"' and (not escape):
                self.pos += 1
                break
            elif self.current_char == '\\':
                escape = True
            else:
                escape = False
            self.pos += 1
        return self.definition[startPos:self.pos]

    def _parse_literal(self) -> ASTLiteral:
        self.skip_ws()
        if self.skip_word('true'):
            return ASTBooleanLiteral(True)
        if self.skip_word('false'):
            return ASTBooleanLiteral(False)
        pos = self.pos
        if self.match(float_literal_re):
            self.match(float_literal_suffix_re)
            return ASTNumberLiteral(self.definition[pos:self.pos])
        for regex in [binary_literal_re, hex_literal_re, integer_literal_re, octal_literal_re]:
            if self.match(regex):
                self.match(integers_literal_suffix_re)
                return ASTNumberLiteral(self.definition[pos:self.pos])
        string = self._parse_string()
        if string is not None:
            return ASTStringLiteral(string)
        if self.match(char_literal_re):
            prefix = self.last_match.group(1)
            data = self.last_match.group(2)
            try:
                return ASTCharLiteral(prefix, data)
            except UnicodeDecodeError as e:
                self.fail('Can not handle character literal. Internal error was: %s' % e)
            except UnsupportedMultiCharacterCharLiteral:
                self.fail('Can not handle character literal resulting in multiple decoded characters.')
        return None

    def _parse_paren_expression(self) -> ASTExpression:
        if self.current_char != '(':
            return None
        self.pos += 1
        res = self._parse_expression()
        self.skip_ws()
        if not self.skip_string(')'):
            self.fail("Expected ')' in end of parenthesized expression.")
        return ASTParenExpr(res)

    def _parse_primary_expression(self) -> ASTExpression:
        self.skip_ws()
        res: ASTExpression = self._parse_literal()
        if res is not None:
            return res
        res = self._parse_paren_expression()
        if res is not None:
            return res
        nn = self._parse_nested_name()
        if nn is not None:
            return ASTIdExpression(nn)
        return None

    def _parse_initializer_list(self, name: str, open: str, close: str) -> Tuple[List[ASTExpression], bool]:
        self.skip_ws()
        if not self.skip_string_and_ws(open):
            return (None, None)
        if self.skip_string(close):
            return ([], False)
        exprs = []
        trailingComma = False
        while True:
            self.skip_ws()
            expr = self._parse_expression()
            self.skip_ws()
            exprs.append(expr)
            self.skip_ws()
            if self.skip_string(close):
                break
            if not self.skip_string_and_ws(','):
                self.fail("Error in %s, expected ',' or '%s'." % (name, close))
            if self.current_char == close and close == '}':
                self.pos += 1
                trailingComma = True
                break
        return (exprs, trailingComma)

    def _parse_paren_expression_list(self) -> ASTParenExprList:
        exprs, trailingComma = self._parse_initializer_list('parenthesized expression-list', '(', ')')
        if exprs is None:
            return None
        return ASTParenExprList(exprs)

    def _parse_braced_init_list(self) -> ASTBracedInitList:
        exprs, trailingComma = self._parse_initializer_list('braced-init-list', '{', '}')
        if exprs is None:
            return None
        return ASTBracedInitList(exprs, trailingComma)

    def _parse_postfix_expression(self) -> ASTPostfixExpr:
        prefix = self._parse_primary_expression()
        postFixes: List[ASTPostfixOp] = []
        while True:
            self.skip_ws()
            if self.skip_string_and_ws('['):
                expr = self._parse_expression()
                self.skip_ws()
                if not self.skip_string(']'):
                    self.fail("Expected ']' in end of postfix expression.")
                postFixes.append(ASTPostfixArray(expr))
                continue
            if self.skip_string('->'):
                if self.skip_string('*'):
                    self.pos -= 3
                else:
                    name = self._parse_nested_name()
                    postFixes.append(ASTPostfixMemberOfPointer(name))
                    continue
            if self.skip_string('++'):
                postFixes.append(ASTPostfixInc())
                continue
            if self.skip_string('--'):
                postFixes.append(ASTPostfixDec())
                continue
            lst = self._parse_paren_expression_list()
            if lst is not None:
                postFixes.append(ASTPostfixCallExpr(lst))
                continue
            break
        return ASTPostfixExpr(prefix, postFixes)

    def _parse_unary_expression(self) -> ASTExpression:
        self.skip_ws()
        for op in _expression_unary_ops:
            if op[0] in 'cn':
                res = self.skip_word(op)
            else:
                res = self.skip_string(op)
            if res:
                expr = self._parse_cast_expression()
                return ASTUnaryOpExpr(op, expr)
        if self.skip_word_and_ws('sizeof'):
            if self.skip_string_and_ws('('):
                typ = self._parse_type(named=False)
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail("Expecting ')' to end 'sizeof'.")
                return ASTSizeofType(typ)
            expr = self._parse_unary_expression()
            return ASTSizeofExpr(expr)
        if self.skip_word_and_ws('alignof'):
            if not self.skip_string_and_ws('('):
                self.fail("Expecting '(' after 'alignof'.")
            typ = self._parse_type(named=False)
            self.skip_ws()
            if not self.skip_string(')'):
                self.fail("Expecting ')' to end 'alignof'.")
            return ASTAlignofExpr(typ)
        return self._parse_postfix_expression()

    def _parse_cast_expression(self) -> ASTExpression:
        pos = self.pos
        self.skip_ws()
        if self.skip_string('('):
            try:
                typ = self._parse_type(False)
                if not self.skip_string(')'):
                    self.fail("Expected ')' in cast expression.")
                expr = self._parse_cast_expression()
                return ASTCastExpr(typ, expr)
            except DefinitionError as exCast:
                self.pos = pos
                try:
                    return self._parse_unary_expression()
                except DefinitionError as exUnary:
                    errs = []
                    errs.append((exCast, 'If type cast expression'))
                    errs.append((exUnary, 'If unary expression'))
                    raise self._make_multi_error(errs, 'Error in cast expression.') from exUnary
        else:
            return self._parse_unary_expression()

    def _parse_logical_or_expression(self) -> ASTExpression:

        def _parse_bin_op_expr(self, opId):
            if opId + 1 == len(_expression_bin_ops):

                def parser() -> ASTExpression:
                    return self._parse_cast_expression()
            else:

                def parser() -> ASTExpression:
                    return _parse_bin_op_expr(self, opId + 1)
            exprs = []
            ops = []
            exprs.append(parser())
            while True:
                self.skip_ws()
                pos = self.pos
                oneMore = False
                for op in _expression_bin_ops[opId]:
                    if op[0] in 'abcnox':
                        if not self.skip_word(op):
                            continue
                    elif not self.skip_string(op):
                        continue
                    if op == '&' and self.current_char == '&':
                        self.pos -= 1
                        break
                    try:
                        expr = parser()
                        exprs.append(expr)
                        ops.append(op)
                        oneMore = True
                        break
                    except DefinitionError:
                        self.pos = pos
                if not oneMore:
                    break
            return ASTBinOpExpr(exprs, ops)
        return _parse_bin_op_expr(self, 0)

    def _parse_conditional_expression_tail(self, orExprHead: Any) -> ASTExpression:
        return None

    def _parse_assignment_expression(self) -> ASTExpression:
        exprs = []
        ops = []
        orExpr = self._parse_logical_or_expression()
        exprs.append(orExpr)
        while True:
            oneMore = False
            self.skip_ws()
            for op in _expression_assignment_ops:
                if op[0] in 'abcnox':
                    if not self.skip_word(op):
                        continue
                elif not self.skip_string(op):
                    continue
                expr = self._parse_logical_or_expression()
                exprs.append(expr)
                ops.append(op)
                oneMore = True
            if not oneMore:
                break
        return ASTAssignmentExpr(exprs, ops)

    def _parse_constant_expression(self) -> ASTExpression:
        orExpr = self._parse_logical_or_expression()
        return orExpr

    def _parse_expression(self) -> ASTExpression:
        return self._parse_assignment_expression()

    def _parse_expression_fallback(self, end: List[str], parser: Callable[[], ASTExpression], allow: bool=True) -> ASTExpression:
        prevPos = self.pos
        try:
            return parser()
        except DefinitionError as e:
            if not allow or not self.allowFallbackExpressionParsing:
                raise
            self.warn('Parsing of expression failed. Using fallback parser. Error was:\n%s' % e)
            self.pos = prevPos
        assert end is not None
        self.skip_ws()
        startPos = self.pos
        if self.match(_string_re):
            value = self.matched_text
        else:
            brackets = {'(': ')', '{': '}', '[': ']'}
            symbols: List[str] = []
            while not self.eof:
                if len(symbols) == 0 and self.current_char in end:
                    break
                if self.current_char in brackets:
                    symbols.append(brackets[self.current_char])
                elif len(symbols) > 0 and self.current_char == symbols[-1]:
                    symbols.pop()
                self.pos += 1
            if len(end) > 0 and self.eof:
                self.fail('Could not find end of expression starting at %d.' % startPos)
            value = self.definition[startPos:self.pos].strip()
        return ASTFallbackExpr(value.strip())

    def _parse_nested_name(self) -> ASTNestedName:
        names: List[Any] = []
        self.skip_ws()
        rooted = False
        if self.skip_string('.'):
            rooted = True
        while 1:
            self.skip_ws()
            if not self.match(identifier_re):
                self.fail('Expected identifier in nested name.')
            identifier = self.matched_text
            if identifier in _keywords:
                self.fail('Expected identifier in nested name, got keyword: %s' % identifier)
            if self.matched_text in self.config.c_extra_keywords:
                msg = 'Expected identifier, got user-defined keyword: %s.' + ' Remove it from c_extra_keywords to allow it as identifier.\n' + 'Currently c_extra_keywords is %s.'
                self.fail(msg % (self.matched_text, str(self.config.c_extra_keywords)))
            ident = ASTIdentifier(identifier)
            names.append(ident)
            self.skip_ws()
            if not self.skip_string('.'):
                break
        return ASTNestedName(names, rooted)

    def _parse_simple_type_specifier(self) -> Optional[str]:
        if self.match(_simple_type_specifiers_re):
            return self.matched_text
        for t in ('bool', 'complex', 'imaginary'):
            if t in self.config.c_extra_keywords:
                if self.skip_word(t):
                    return t
        return None

    def _parse_simple_type_specifiers(self) -> ASTTrailingTypeSpecFundamental:
        names: List[str] = []
        self.skip_ws()
        while True:
            t = self._parse_simple_type_specifier()
            if t is None:
                break
            names.append(t)
            self.skip_ws()
        if len(names) == 0:
            return None
        return ASTTrailingTypeSpecFundamental(names)

    def _parse_trailing_type_spec(self) -> ASTTrailingTypeSpec:
        self.skip_ws()
        res = self._parse_simple_type_specifiers()
        if res is not None:
            return res
        prefix = None
        self.skip_ws()
        for k in ('struct', 'enum', 'union'):
            if self.skip_word_and_ws(k):
                prefix = k
                break
        nestedName = self._parse_nested_name()
        return ASTTrailingTypeSpecName(prefix, nestedName)

    def _parse_parameters(self, paramMode: str) -> Optional[ASTParameters]:
        self.skip_ws()
        if not self.skip_string('('):
            if paramMode == 'function':
                self.fail('Expecting "(" in parameters.')
            else:
                return None
        args = []
        self.skip_ws()
        if not self.skip_string(')'):
            while 1:
                self.skip_ws()
                if self.skip_string('...'):
                    args.append(ASTFunctionParameter(None, True))
                    self.skip_ws()
                    if not self.skip_string(')'):
                        self.fail('Expected ")" after "..." in parameters.')
                    break
                arg = self._parse_type_with_init(outer=None, named='single')
                args.append(ASTFunctionParameter(arg))
                self.skip_ws()
                if self.skip_string(','):
                    continue
                elif self.skip_string(')'):
                    break
                else:
                    self.fail('Expecting "," or ")" in parameters, got "%s".' % self.current_char)
        attrs = self._parse_attribute_list()
        return ASTParameters(args, attrs)

    def _parse_decl_specs_simple(self, outer: str, typed: bool) -> ASTDeclSpecsSimple:
        """Just parse the simple ones."""
        storage = None
        threadLocal = None
        inline = None
        restrict = None
        volatile = None
        const = None
        attrs = []
        while 1:
            self.skip_ws()
            if not storage:
                if outer == 'member':
                    if self.skip_word('auto'):
                        storage = 'auto'
                        continue
                    if self.skip_word('register'):
                        storage = 'register'
                        continue
                if outer in ('member', 'function'):
                    if self.skip_word('static'):
                        storage = 'static'
                        continue
                    if self.skip_word('extern'):
                        storage = 'extern'
                        continue
            if outer == 'member' and (not threadLocal):
                if self.skip_word('thread_local'):
                    threadLocal = 'thread_local'
                    continue
                if self.skip_word('_Thread_local'):
                    threadLocal = '_Thread_local'
                    continue
            if outer == 'function' and (not inline):
                inline = self.skip_word('inline')
                if inline:
                    continue
            if not restrict and typed:
                restrict = self.skip_word('restrict')
                if restrict:
                    continue
            if not volatile and typed:
                volatile = self.skip_word('volatile')
                if volatile:
                    continue
            if not const and typed:
                const = self.skip_word('const')
                if const:
                    continue
            attr = self._parse_attribute()
            if attr:
                attrs.append(attr)
                continue
            break
        return ASTDeclSpecsSimple(storage, threadLocal, inline, restrict, volatile, const, ASTAttributeList(attrs))

    def _parse_decl_specs(self, outer: str, typed: bool=True) -> ASTDeclSpecs:
        if outer:
            if outer not in ('type', 'member', 'function'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
        leftSpecs = self._parse_decl_specs_simple(outer, typed)
        rightSpecs = None
        if typed:
            trailing = self._parse_trailing_type_spec()
            rightSpecs = self._parse_decl_specs_simple(outer, typed)
        else:
            trailing = None
        return ASTDeclSpecs(outer, leftSpecs, rightSpecs, trailing)

    def _parse_declarator_name_suffix(self, named: Union[bool, str], paramMode: str, typed: bool) -> ASTDeclarator:
        assert named in (True, False, 'single')
        if named == 'single':
            if self.match(identifier_re):
                if self.matched_text in _keywords:
                    self.fail('Expected identifier, got keyword: %s' % self.matched_text)
                if self.matched_text in self.config.c_extra_keywords:
                    msg = 'Expected identifier, got user-defined keyword: %s.' + ' Remove it from c_extra_keywords to allow it as identifier.\n' + 'Currently c_extra_keywords is %s.'
                    self.fail(msg % (self.matched_text, str(self.config.c_extra_keywords)))
                identifier = ASTIdentifier(self.matched_text)
                declId = ASTNestedName([identifier], rooted=False)
            else:
                declId = None
        elif named:
            declId = self._parse_nested_name()
        else:
            declId = None
        arrayOps = []
        while 1:
            self.skip_ws()
            if typed and self.skip_string('['):
                self.skip_ws()
                static = False
                const = False
                volatile = False
                restrict = False
                while True:
                    if not static:
                        if self.skip_word_and_ws('static'):
                            static = True
                            continue
                    if not const:
                        if self.skip_word_and_ws('const'):
                            const = True
                            continue
                    if not volatile:
                        if self.skip_word_and_ws('volatile'):
                            volatile = True
                            continue
                    if not restrict:
                        if self.skip_word_and_ws('restrict'):
                            restrict = True
                            continue
                    break
                vla = False if static else self.skip_string_and_ws('*')
                if vla:
                    if not self.skip_string(']'):
                        self.fail("Expected ']' in end of array operator.")
                    size = None
                elif self.skip_string(']'):
                    size = None
                else:

                    def parser():
                        return self._parse_expression()
                    size = self._parse_expression_fallback([']'], parser)
                    self.skip_ws()
                    if not self.skip_string(']'):
                        self.fail("Expected ']' in end of array operator.")
                arrayOps.append(ASTArray(static, const, volatile, restrict, vla, size))
            else:
                break
        param = self._parse_parameters(paramMode)
        if param is None and len(arrayOps) == 0:
            if named and paramMode == 'type' and typed:
                self.skip_ws()
                if self.skip_string(':'):
                    size = self._parse_constant_expression()
                    return ASTDeclaratorNameBitField(declId=declId, size=size)
        return ASTDeclaratorNameParam(declId=declId, arrayOps=arrayOps, param=param)

    def _parse_declarator(self, named: Union[bool, str], paramMode: str, typed: bool=True) -> ASTDeclarator:
        if paramMode not in ('type', 'function'):
            raise Exception("Internal error, unknown paramMode '%s'." % paramMode)
        prevErrors = []
        self.skip_ws()
        if typed and self.skip_string('*'):
            self.skip_ws()
            restrict = False
            volatile = False
            const = False
            attrs = []
            while 1:
                if not restrict:
                    restrict = self.skip_word_and_ws('restrict')
                    if restrict:
                        continue
                if not volatile:
                    volatile = self.skip_word_and_ws('volatile')
                    if volatile:
                        continue
                if not const:
                    const = self.skip_word_and_ws('const')
                    if const:
                        continue
                attr = self._parse_attribute()
                if attr is not None:
                    attrs.append(attr)
                    continue
                break
            next = self._parse_declarator(named, paramMode, typed)
            return ASTDeclaratorPtr(next=next, restrict=restrict, volatile=volatile, const=const, attrs=ASTAttributeList(attrs))
        if typed and self.current_char == '(':
            pos = self.pos
            try:
                res = self._parse_declarator_name_suffix(named, paramMode, typed)
                return res
            except DefinitionError as exParamQual:
                msg = 'If declarator-id with parameters'
                if paramMode == 'function':
                    msg += " (e.g., 'void f(int arg)')"
                prevErrors.append((exParamQual, msg))
                self.pos = pos
                try:
                    assert self.current_char == '('
                    self.skip_string('(')
                    inner = self._parse_declarator(named, paramMode, typed)
                    if not self.skip_string(')'):
                        self.fail('Expected \')\' in "( ptr-declarator )"')
                    next = self._parse_declarator(named=False, paramMode='type', typed=typed)
                    return ASTDeclaratorParen(inner=inner, next=next)
                except DefinitionError as exNoPtrParen:
                    self.pos = pos
                    msg = 'If parenthesis in noptr-declarator'
                    if paramMode == 'function':
                        msg += " (e.g., 'void (*f(int arg))(double)')"
                    prevErrors.append((exNoPtrParen, msg))
                    header = 'Error in declarator'
                    raise self._make_multi_error(prevErrors, header) from exNoPtrParen
        pos = self.pos
        try:
            return self._parse_declarator_name_suffix(named, paramMode, typed)
        except DefinitionError as e:
            self.pos = pos
            prevErrors.append((e, 'If declarator-id'))
            header = 'Error in declarator or parameters'
            raise self._make_multi_error(prevErrors, header) from e

    def _parse_initializer(self, outer: str=None, allowFallback: bool=True) -> ASTInitializer:
        self.skip_ws()
        if outer == 'member' and False:
            bracedInit = self._parse_braced_init_list()
            if bracedInit is not None:
                return ASTInitializer(bracedInit, hasAssign=False)
        if not self.skip_string('='):
            return None
        bracedInit = self._parse_braced_init_list()
        if bracedInit is not None:
            return ASTInitializer(bracedInit)
        if outer == 'member':
            fallbackEnd: List[str] = []
        elif outer is None:
            fallbackEnd = [',', ')']
        else:
            self.fail("Internal error, initializer for outer '%s' not implemented." % outer)

        def parser():
            return self._parse_assignment_expression()
        value = self._parse_expression_fallback(fallbackEnd, parser, allow=allowFallback)
        return ASTInitializer(value)

    def _parse_type(self, named: Union[bool, str], outer: Optional[str]=None) -> ASTType:
        """
        named=False|'single'|True: 'single' is e.g., for function objects which
        doesn't need to name the arguments, but otherwise is a single name
        """
        if outer:
            if outer not in ('type', 'member', 'function'):
                raise Exception('Internal error, unknown outer "%s".' % outer)
            assert named
        if outer == 'type':
            prevErrors = []
            startPos = self.pos
            try:
                declSpecs = self._parse_decl_specs(outer=outer, typed=False)
                decl = self._parse_declarator(named=True, paramMode=outer, typed=False)
                self.assert_end(allowSemicolon=True)
            except DefinitionError as exUntyped:
                desc = 'If just a name'
                prevErrors.append((exUntyped, desc))
                self.pos = startPos
                try:
                    declSpecs = self._parse_decl_specs(outer=outer)
                    decl = self._parse_declarator(named=True, paramMode=outer)
                except DefinitionError as exTyped:
                    self.pos = startPos
                    desc = 'If typedef-like declaration'
                    prevErrors.append((exTyped, desc))
                    if True:
                        header = 'Type must be either just a name or a '
                        header += 'typedef-like declaration.'
                        raise self._make_multi_error(prevErrors, header) from exTyped
                    else:
                        self.pos = startPos
                        typed = True
                        declSpecs = self._parse_decl_specs(outer=outer, typed=typed)
                        decl = self._parse_declarator(named=True, paramMode=outer, typed=typed)
        elif outer == 'function':
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=True, paramMode=outer)
        else:
            paramMode = 'type'
            if outer == 'member':
                named = True
            declSpecs = self._parse_decl_specs(outer=outer)
            decl = self._parse_declarator(named=named, paramMode=paramMode)
        return ASTType(declSpecs, decl)

    def _parse_type_with_init(self, named: Union[bool, str], outer: str) -> ASTTypeWithInit:
        if outer:
            assert outer in ('type', 'member', 'function')
        type = self._parse_type(outer=outer, named=named)
        init = self._parse_initializer(outer=outer)
        return ASTTypeWithInit(type, init)

    def _parse_macro(self) -> ASTMacro:
        self.skip_ws()
        ident = self._parse_nested_name()
        if ident is None:
            self.fail('Expected identifier in macro definition.')
        self.skip_ws()
        if not self.skip_string_and_ws('('):
            return ASTMacro(ident, None)
        if self.skip_string(')'):
            return ASTMacro(ident, [])
        args = []
        while 1:
            self.skip_ws()
            if self.skip_string('...'):
                args.append(ASTMacroParameter(None, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            if not self.match(identifier_re):
                self.fail('Expected identifier in macro parameters.')
            nn = ASTNestedName([ASTIdentifier(self.matched_text)], rooted=False)
            self.skip_ws()
            if self.skip_string_and_ws('...'):
                args.append(ASTMacroParameter(nn, False, True))
                self.skip_ws()
                if not self.skip_string(')'):
                    self.fail('Expected ")" after "..." in macro parameters.')
                break
            args.append(ASTMacroParameter(nn))
            if self.skip_string_and_ws(','):
                continue
            elif self.skip_string_and_ws(')'):
                break
            else:
                self.fail("Expected identifier, ')', or ',' in macro parameter list.")
        return ASTMacro(ident, args)

    def _parse_struct(self) -> ASTStruct:
        name = self._parse_nested_name()
        return ASTStruct(name)

    def _parse_union(self) -> ASTUnion:
        name = self._parse_nested_name()
        return ASTUnion(name)

    def _parse_enum(self) -> ASTEnum:
        name = self._parse_nested_name()
        return ASTEnum(name)

    def _parse_enumerator(self) -> ASTEnumerator:
        name = self._parse_nested_name()
        attrs = self._parse_attribute_list()
        self.skip_ws()
        init = None
        if self.skip_string('='):
            self.skip_ws()

            def parser() -> ASTExpression:
                return self._parse_constant_expression()
            initVal = self._parse_expression_fallback([], parser)
            init = ASTInitializer(initVal)
        return ASTEnumerator(name, init, attrs)

    def parse_pre_v3_type_definition(self) -> ASTDeclaration:
        self.skip_ws()
        declaration: DeclarationType = None
        if self.skip_word('struct'):
            typ = 'struct'
            declaration = self._parse_struct()
        elif self.skip_word('union'):
            typ = 'union'
            declaration = self._parse_union()
        elif self.skip_word('enum'):
            typ = 'enum'
            declaration = self._parse_enum()
        else:
            self.fail("Could not parse pre-v3 type directive. Must start with 'struct', 'union', or 'enum'.")
        return ASTDeclaration(typ, typ, declaration, False)

    def parse_declaration(self, objectType: str, directiveType: str) -> ASTDeclaration:
        if objectType not in ('function', 'member', 'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown objectType "%s".' % objectType)
        if directiveType not in ('function', 'member', 'var', 'macro', 'struct', 'union', 'enum', 'enumerator', 'type'):
            raise Exception('Internal error, unknown directiveType "%s".' % directiveType)
        declaration: DeclarationType = None
        if objectType == 'member':
            declaration = self._parse_type_with_init(named=True, outer='member')
        elif objectType == 'function':
            declaration = self._parse_type(named=True, outer='function')
        elif objectType == 'macro':
            declaration = self._parse_macro()
        elif objectType == 'struct':
            declaration = self._parse_struct()
        elif objectType == 'union':
            declaration = self._parse_union()
        elif objectType == 'enum':
            declaration = self._parse_enum()
        elif objectType == 'enumerator':
            declaration = self._parse_enumerator()
        elif objectType == 'type':
            declaration = self._parse_type(named=True, outer='type')
        else:
            raise AssertionError()
        if objectType != 'macro':
            self.skip_ws()
            semicolon = self.skip_string(';')
        else:
            semicolon = False
        return ASTDeclaration(objectType, directiveType, declaration, semicolon)

    def parse_namespace_object(self) -> ASTNestedName:
        return self._parse_nested_name()

    def parse_xref_object(self) -> ASTNestedName:
        name = self._parse_nested_name()
        self.skip_ws()
        self.skip_string('()')
        self.assert_end()
        return name

    def parse_expression(self) -> Union[ASTExpression, ASTType]:
        pos = self.pos
        res: Union[ASTExpression, ASTType] = None
        try:
            res = self._parse_expression()
            self.skip_ws()
            self.assert_end()
        except DefinitionError as exExpr:
            self.pos = pos
            try:
                res = self._parse_type(False)
                self.skip_ws()
                self.assert_end()
            except DefinitionError as exType:
                header = 'Error when parsing (type) expression.'
                errs = []
                errs.append((exExpr, 'If expression'))
                errs.append((exType, 'If type'))
                raise self._make_multi_error(errs, header) from exType
        return res