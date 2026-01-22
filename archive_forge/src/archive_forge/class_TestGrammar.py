from __future__ import absolute_import
import ast
import textwrap
from ...TestUtils import CythonTest
from .. import ExprNodes
from ..Errors import CompileError
class TestGrammar(CythonTest):

    def test_invalid_number_literals(self):
        for literal in INVALID_UNDERSCORE_LITERALS:
            for expression in ['%s', '1 + %s', '%s + 1', '2 * %s', '%s * 2']:
                code = 'x = ' + expression % literal
                try:
                    self.fragment(u'                    # cython: language_level=3\n                    ' + code)
                except CompileError as exc:
                    assert code in [s.strip() for s in str(exc).splitlines()], str(exc)
                else:
                    assert False, "Invalid Cython code '%s' failed to raise an exception" % code

    def test_valid_number_literals(self):
        for literal in VALID_UNDERSCORE_LITERALS:
            for i, expression in enumerate(['%s', '1 + %s', '%s + 1', '2 * %s', '%s * 2']):
                code = 'x = ' + expression % literal
                node = self.fragment(u'                    # cython: language_level=3\n                    ' + code).root
                assert node is not None
                literal_node = node.stats[0].rhs
                if i > 0:
                    literal_node = literal_node.operand2 if i % 2 else literal_node.operand1
                if 'j' in literal or 'J' in literal:
                    if '+' in literal:
                        assert isinstance(literal_node, ExprNodes.AddNode), (literal, literal_node)
                    else:
                        assert isinstance(literal_node, ExprNodes.ImagNode), (literal, literal_node)
                elif '.' in literal or 'e' in literal or ('E' in literal and (not ('0x' in literal or '0X' in literal))):
                    assert isinstance(literal_node, ExprNodes.FloatNode), (literal, literal_node)
                else:
                    assert isinstance(literal_node, ExprNodes.IntNode), (literal, literal_node)

    def test_invalid_ellipsis(self):
        ERR = ':{0}:{1}: Expected an identifier or literal'
        for code, line, col in INVALID_ELLIPSIS:
            try:
                ast.parse(textwrap.dedent(code))
            except SyntaxError as exc:
                assert True
            else:
                assert False, "Invalid Python code '%s' failed to raise an exception" % code
            try:
                self.fragment(u'                # cython: language_level=3\n                ' + code)
            except CompileError as exc:
                assert ERR.format(line, col) in str(exc), str(exc)
            else:
                assert False, "Invalid Cython code '%s' failed to raise an exception" % code