import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class InputSplitterTestCase(unittest.TestCase):

    def setUp(self):
        self.isp = isp.InputSplitter()

    def test_reset(self):
        isp = self.isp
        isp.push('x=1')
        isp.reset()
        self.assertEqual(isp._buffer, [])
        self.assertEqual(isp.get_indent_spaces(), 0)
        self.assertEqual(isp.source, '')
        self.assertEqual(isp.code, None)
        self.assertEqual(isp._is_complete, False)

    def test_source(self):
        self.isp._store('1')
        self.isp._store('2')
        self.assertEqual(self.isp.source, '1\n2\n')
        self.assertEqual(len(self.isp._buffer) > 0, True)
        self.assertEqual(self.isp.source_reset(), '1\n2\n')
        self.assertEqual(self.isp._buffer, [])
        self.assertEqual(self.isp.source, '')

    def test_indent(self):
        isp = self.isp
        isp.push('x=1')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_indent2(self):
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push(' ' * 2)
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent3(self):
        isp = self.isp
        isp.push('if 1:')
        isp.push('    x = (1+\n    2)')
        self.assertEqual(isp.get_indent_spaces(), 4)

    def test_indent4(self):
        isp = self.isp
        isp.push('if 1: \n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\t\n    x=1')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('y=2\n')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_pass(self):
        isp = self.isp
        isp.push('if 1:\n    passes = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     pass')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     pass   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_break(self):
        isp = self.isp
        isp.push('while 1:\n    breaks = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     break')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     break   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_continue(self):
        isp = self.isp
        isp.push('while 1:\n    continues = 5')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('while 1:\n     continue')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('while 1:\n     continue   ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_raise(self):
        isp = self.isp
        isp.push('if 1:\n    raised = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     raise TypeError()')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     raise      ')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_dedent_return(self):
        isp = self.isp
        isp.push('if 1:\n    returning = 4')
        self.assertEqual(isp.get_indent_spaces(), 4)
        isp.push('if 1:\n     return 5 + 493')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return      ')
        self.assertEqual(isp.get_indent_spaces(), 0)
        isp.push('if 1:\n     return(0)')
        self.assertEqual(isp.get_indent_spaces(), 0)

    def test_push(self):
        isp = self.isp
        self.assertEqual(isp.push('x=1'), True)

    def test_push2(self):
        isp = self.isp
        self.assertEqual(isp.push('if 1:'), False)
        for line in ['  x=1', '# a comment', '  y=2']:
            print(line)
            self.assertEqual(isp.push(line), True)

    def test_push3(self):
        isp = self.isp
        isp.push('if True:')
        isp.push('  a = 1')
        self.assertEqual(isp.push('b = [1,'), False)

    def test_push_accepts_more(self):
        isp = self.isp
        isp.push('x=1')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more2(self):
        isp = self.isp
        isp.push('if 1:')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('  x=1')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more3(self):
        isp = self.isp
        isp.push('x = (2+\n3)')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more4(self):
        isp = self.isp
        isp.push('if 1:')
        isp.push('    x = (2+')
        isp.push('    3)')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('    y = 3')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_push_accepts_more5(self):
        isp = self.isp
        isp.push('try:')
        isp.push('    a = 5')
        isp.push('except:')
        isp.push('    raise')
        self.assertEqual(isp.push_accepts_more(), True)

    def test_continuation(self):
        isp = self.isp
        isp.push('import os, \\')
        self.assertEqual(isp.push_accepts_more(), True)
        isp.push('sys')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_syntax_error(self):
        isp = self.isp
        isp.push('run foo')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_unicode(self):
        self.isp.push(u'Pérez')
        self.isp.push(u'Ã©')
        self.isp.push(u"u'Ã©'")

    @pytest.mark.xfail(reason='Bug in python 3.9.8 –\xa0bpo 45738', condition=sys.version_info in [(3, 11, 0, 'alpha', 2)], raises=SystemError, strict=True)
    def test_line_continuation(self):
        """ Test issue #2108."""
        isp = self.isp
        isp.push('1 \\\n\n')
        self.assertEqual(isp.push_accepts_more(), False)
        isp.push('1 \\ ')
        self.assertEqual(isp.push_accepts_more(), False)
        isp.push('(1 \\ ')
        self.assertEqual(isp.push_accepts_more(), False)

    def test_check_complete(self):
        isp = self.isp
        self.assertEqual(isp.check_complete('a = 1'), ('complete', None))
        self.assertEqual(isp.check_complete('for a in range(5):'), ('incomplete', 4))
        self.assertEqual(isp.check_complete('raise = 2'), ('invalid', None))
        self.assertEqual(isp.check_complete('a = [1,\n2,'), ('incomplete', 0))
        self.assertEqual(isp.check_complete('def a():\n x=1\n global x'), ('invalid', None))