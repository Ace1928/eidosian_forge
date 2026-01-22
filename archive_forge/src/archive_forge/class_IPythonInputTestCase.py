import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class IPythonInputTestCase(InputSplitterTestCase):
    """By just creating a new class whose .isp is a different instance, we
    re-run the same test battery on the new input splitter.

    In addition, this runs the tests over the syntax and syntax_ml dicts that
    were tested by individual functions, as part of the OO interface.

    It also makes some checks on the raw buffer storage.
    """

    def setUp(self):
        self.isp = isp.IPythonInputSplitter()

    def test_syntax(self):
        """Call all single-line syntax tests from the main object"""
        isp = self.isp
        for example in syntax.values():
            for raw, out_t in example:
                if raw.startswith(' '):
                    continue
                isp.push(raw + '\n')
                out_raw = isp.source_raw
                out = isp.source_reset()
                self.assertEqual(out.rstrip(), out_t, tt.pair_fail_msg.format('inputsplitter', raw, out_t, out))
                self.assertEqual(out_raw.rstrip(), raw.rstrip())

    def test_syntax_multiline(self):
        isp = self.isp
        for example in syntax_ml.values():
            for line_pairs in example:
                out_t_parts = []
                raw_parts = []
                for lraw, out_t_part in line_pairs:
                    if out_t_part is not None:
                        out_t_parts.append(out_t_part)
                    if lraw is not None:
                        isp.push(lraw)
                        raw_parts.append(lraw)
                out_raw = isp.source_raw
                out = isp.source_reset()
                out_t = '\n'.join(out_t_parts).rstrip()
                raw = '\n'.join(raw_parts).rstrip()
                self.assertEqual(out.rstrip(), out_t)
                self.assertEqual(out_raw.rstrip(), raw)

    def test_syntax_multiline_cell(self):
        isp = self.isp
        for example in syntax_ml.values():
            out_t_parts = []
            for line_pairs in example:
                raw = '\n'.join((r for r, _ in line_pairs if r is not None))
                out_t = '\n'.join((t for _, t in line_pairs if t is not None))
                out = isp.transform_cell(raw)
                self.assertEqual(out.rstrip(), out_t.rstrip())

    def test_cellmagic_preempt(self):
        isp = self.isp
        for raw, name, line, cell in [('%%cellm a\nIn[1]:', u'cellm', u'a', u'In[1]:'), ('%%cellm \nline\n>>> hi', u'cellm', u'', u'line\n>>> hi'), ('>>> %%cellm \nline\n>>> hi', u'cellm', u'', u'line\nhi'), ('%%cellm \n>>> hi', u'cellm', u'', u'>>> hi'), ('%%cellm \nline1\nline2', u'cellm', u'', u'line1\nline2'), ('%%cellm \nline1\\\\\nline2', u'cellm', u'', u'line1\\\\\nline2')]:
            expected = 'get_ipython().run_cell_magic(%r, %r, %r)' % (name, line, cell)
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())

    def test_multiline_passthrough(self):
        isp = self.isp

        class CommentTransformer(InputTransformer):

            def __init__(self):
                self._lines = []

            def push(self, line):
                self._lines.append(line + '#')

            def reset(self):
                text = '\n'.join(self._lines)
                self._lines = []
                return text
        isp.physical_line_transforms.insert(0, CommentTransformer())
        for raw, expected in [('a=5', 'a=5#'), ('%ls foo', 'get_ipython().run_line_magic(%r, %r)' % (u'ls', u'foo#')), ('!ls foo\n%ls bar', 'get_ipython().system(%r)\nget_ipython().run_line_magic(%r, %r)' % (u'ls foo#', u'ls', u'bar#')), ('1\n2\n3\n%ls foo\n4\n5', '1#\n2#\n3#\nget_ipython().run_line_magic(%r, %r)\n4#\n5#' % (u'ls', u'foo#'))]:
            out = isp.transform_cell(raw)
            self.assertEqual(out.rstrip(), expected.rstrip())