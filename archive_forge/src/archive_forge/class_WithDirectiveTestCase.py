import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class WithDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:with` template directive."""

    def test_shadowing(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          ${x}\n          <span py:with="x = x * 2" py:replace="x"/>\n          ${x}\n        </div>')
        self.assertEqual('<div>\n          42\n          84\n          42\n        </div>', tmpl.generate(x=42).render(encoding=None))

    def test_as_element(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = x * 2">${x}</py:with>\n        </div>')
        self.assertEqual('<div>\n          84\n        </div>', tmpl.generate(x=42).render(encoding=None))

    def test_multiple_vars_same_name(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="\n            foo = \'bar\';\n            foo = foo.replace(\'r\', \'z\')\n          ">\n            $foo\n          </py:with>\n        </div>')
        self.assertEqual('<div>\n            baz\n        </div>', tmpl.generate(x=42).render(encoding=None))

    def test_multiple_vars_single_assignment(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = y = z = 1">${x} ${y} ${z}</py:with>\n        </div>')
        self.assertEqual('<div>\n          1 1 1\n        </div>', tmpl.generate(x=42).render(encoding=None))

    def test_nested_vars_single_assignment(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x, (y, z) = (1, (2, 3))">${x} ${y} ${z}</py:with>\n        </div>')
        self.assertEqual('<div>\n          1 2 3\n        </div>', tmpl.generate(x=42).render(encoding=None))

    def test_multiple_vars_trailing_semicolon(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = x * 2; y = x / 2;">${x} ${y}</py:with>\n        </div>')
        self.assertEqual('<div>\n          84 %s\n        </div>' % (84 / 2), tmpl.generate(x=42).render(encoding=None))

    def test_semicolon_escape(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:with vars="x = \'here is a semicolon: ;\'; y = \'here are two semicolons: ;;\' ;">\n            ${x}\n            ${y}\n          </py:with>\n        </div>')
        self.assertEqual('<div>\n            here is a semicolon: ;\n            here are two semicolons: ;;\n        </div>', tmpl.generate().render(encoding=None))

    def test_ast_transformation(self):
        """
        Verify that the usual template expression AST transformations are
        applied despite the code being compiled to a `Suite` object.
        """
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="bar=foo.bar">\n            $bar\n          </span>\n        </div>')
        self.assertEqual('<div>\n          <span>\n            42\n          </span>\n        </div>', tmpl.generate(foo={'bar': 42}).render(encoding=None))

    def test_unicode_expr(self):
        tmpl = MarkupTemplate(u'<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="weeks=(u\'一\', u\'二\', u\'三\', u\'四\', u\'五\', u\'六\', u\'日\')">\n            $weeks\n          </span>\n        </div>')
        self.assertEqual(u'<div>\n          <span>\n            一二三四五六日\n          </span>\n        </div>', tmpl.generate().render(encoding=None))

    def test_with_empty_value(self):
        """
        Verify that an empty py:with works (useless, but legal)
        """
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span py:with="">Text</span></div>')
        self.assertEqual('<div>\n          <span>Text</span></div>', tmpl.generate().render(encoding=None))