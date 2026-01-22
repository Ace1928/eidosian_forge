import doctest
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import six
from genshi.compat import BytesIO, StringIO
from genshi.core import Markup
from genshi.filters.i18n import Translator
from genshi.input import XML
from genshi.template.base import BadDirectiveError, TemplateSyntaxError
from genshi.template.loader import TemplateLoader, TemplateNotFound
from genshi.template.markup import MarkupTemplate
class MarkupTemplateTestCase(unittest.TestCase):
    """Tests for markup template processing."""

    def test_parse_fileobj(self):
        fileobj = StringIO('<root> ${var} $var</root>')
        tmpl = MarkupTemplate(fileobj)
        self.assertEqual('<root> 42 42</root>', str(tmpl.generate(var=42)))

    def test_parse_stream(self):
        stream = XML('<root> ${var} $var</root>')
        tmpl = MarkupTemplate(stream)
        self.assertEqual('<root> 42 42</root>', str(tmpl.generate(var=42)))

    def test_pickle(self):
        stream = XML('<root>$var</root>')
        tmpl = MarkupTemplate(stream)
        buf = BytesIO()
        pickle.dump(tmpl, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        self.assertEqual('<root>42</root>', str(unpickled.generate(var=42)))

    def test_interpolate_mixed3(self):
        tmpl = MarkupTemplate('<root> ${var} $var</root>')
        self.assertEqual('<root> 42 42</root>', str(tmpl.generate(var=42)))

    def test_interpolate_leading_trailing_space(self):
        tmpl = MarkupTemplate('<root>${    foo    }</root>')
        self.assertEqual('<root>bar</root>', str(tmpl.generate(foo='bar')))

    def test_interpolate_multiline(self):
        tmpl = MarkupTemplate("<root>${dict(\n          bar = 'baz'\n        )[foo]}</root>")
        self.assertEqual('<root>baz</root>', str(tmpl.generate(foo='bar')))

    def test_interpolate_non_string_attrs(self):
        tmpl = MarkupTemplate('<root attr="${1}"/>')
        self.assertEqual('<root attr="1"/>', str(tmpl.generate()))

    def test_interpolate_list_result(self):
        tmpl = MarkupTemplate('<root>$foo</root>')
        self.assertEqual('<root>buzz</root>', str(tmpl.generate(foo=('buzz',))))

    def test_empty_attr(self):
        tmpl = MarkupTemplate('<root attr=""/>')
        self.assertEqual('<root attr=""/>', str(tmpl.generate()))

    def test_empty_attr_interpolated(self):
        tmpl = MarkupTemplate('<root attr="$attr"/>')
        self.assertEqual('<root attr=""/>', str(tmpl.generate(attr='')))

    def test_bad_directive_error(self):
        xml = '<p xmlns:py="http://genshi.edgewall.org/" py:do="nothing" />'
        try:
            tmpl = MarkupTemplate(xml, filename='test.html')
        except BadDirectiveError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(1, e.lineno)

    def test_directive_value_syntax_error(self):
        xml = '<p xmlns:py="http://genshi.edgewall.org/" py:if="bar\'" />'
        try:
            tmpl = MarkupTemplate(xml, filename='test.html').generate()
            self.fail('Expected TemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(1, e.lineno)

    def test_expression_syntax_error(self):
        xml = '<p>\n          Foo <em>${bar"}</em>\n        </p>'
        try:
            tmpl = MarkupTemplate(xml, filename='test.html')
            self.fail('Expected TemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(2, e.lineno)

    def test_expression_syntax_error_multi_line(self):
        xml = '<p><em></em>\n\n ${bar"}\n\n        </p>'
        try:
            tmpl = MarkupTemplate(xml, filename='test.html')
            self.fail('Expected TemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            self.assertEqual(3, e.lineno)

    def test_markup_noescape(self):
        """
        Verify that outputting context data that is a `Markup` instance is not
        escaped.
        """
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          $myvar\n        </div>')
        self.assertEqual('<div>\n          <b>foo</b>\n        </div>', str(tmpl.generate(myvar=Markup('<b>foo</b>'))))

    def test_text_noescape_quotes(self):
        """
        Verify that outputting context data in text nodes doesn't escape
        quotes.
        """
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          $myvar\n        </div>')
        self.assertEqual('<div>\n          "foo"\n        </div>', str(tmpl.generate(myvar='"foo"')))

    def test_attr_escape_quotes(self):
        """
        Verify that outputting context data in attribtes escapes quotes.
        """
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <elem class="$myvar"/>\n        </div>')
        self.assertEqual('<div>\n          <elem class="&#34;foo&#34;"/>\n        </div>', str(tmpl.generate(myvar='"foo"')))

    def test_directive_element(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <py:if test="myvar">bar</py:if>\n        </div>')
        self.assertEqual('<div>\n          bar\n        </div>', str(tmpl.generate(myvar='"foo"')))

    def test_normal_comment(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <!-- foo bar -->\n        </div>')
        self.assertEqual('<div>\n          <!-- foo bar -->\n        </div>', str(tmpl.generate()))

    def test_template_comment(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <!-- !foo -->\n          <!--!bar-->\n        </div>')
        self.assertEqual('<div>\n        </div>', str(tmpl.generate()))

    def test_parse_with_same_namespace_nested(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n          <span xmlns:py="http://genshi.edgewall.org/">\n          </span>\n        </div>')
        self.assertEqual('<div>\n          <span>\n          </span>\n        </div>', str(tmpl.generate()))

    def test_latin1_encoded_with_xmldecl(self):
        tmpl = MarkupTemplate(u'<?xml version="1.0" encoding="iso-8859-1" ?>\n        <div xmlns:py="http://genshi.edgewall.org/">\n          ö\n        </div>'.encode('iso-8859-1'), encoding='iso-8859-1')
        self.assertEqual(u'<?xml version="1.0" encoding="iso-8859-1"?>\n<div>\n          ö\n        </div>', six.text_type(tmpl.generate()))

    def test_latin1_encoded_explicit_encoding(self):
        tmpl = MarkupTemplate(u'<div xmlns:py="http://genshi.edgewall.org/">\n          ö\n        </div>'.encode('iso-8859-1'), encoding='iso-8859-1')
        self.assertEqual(u'<div>\n          ö\n        </div>', six.text_type(tmpl.generate()))

    def test_exec_with_trailing_space(self):
        """
        Verify that a code block processing instruction with trailing space
        does not cause a syntax error (see ticket #127).
        """
        MarkupTemplate('<foo>\n          <?python\n            bar = 42\n          ?>\n        </foo>')

    def test_exec_import(self):
        tmpl = MarkupTemplate('<?python from datetime import timedelta ?>\n        <div xmlns:py="http://genshi.edgewall.org/">\n          ${timedelta(days=2)}\n        </div>')
        self.assertEqual('<div>\n          2 days, 0:00:00\n        </div>', str(tmpl.generate()))

    def test_exec_def(self):
        tmpl = MarkupTemplate('\n        <?python\n        def foo():\n            return 42\n        ?>\n        <div xmlns:py="http://genshi.edgewall.org/">\n          ${foo()}\n        </div>')
        self.assertEqual('<div>\n          42\n        </div>', str(tmpl.generate()))

    def test_namespace_on_removed_elem(self):
        """
        Verify that a namespace declaration on an element that is removed from
        the generated stream does not get pushed up to the next non-stripped
        element (see ticket #107).
        """
        tmpl = MarkupTemplate('<?xml version="1.0"?>\n        <Test xmlns:py="http://genshi.edgewall.org/">\n          <Size py:if="0" xmlns:t="test">Size</Size>\n          <Item/>\n        </Test>')
        self.assertEqual('<?xml version="1.0"?>\n<Test>\n          \n          <Item/>\n        </Test>', str(tmpl.generate()))

    def test_include_in_loop(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included $idx</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="${name}.html" py:for="idx in range(3)" />\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  <div>Included 0</div><div>Included 1</div><div>Included 2</div>\n                </html>', tmpl.generate(name='tmpl1').render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_dynamic_include_href(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="${name}.html" />\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  <div>Included</div>\n                </html>', tmpl.generate(name='tmpl1').render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_select_included_elements(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<li>$item</li>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <ul py:match="ul">${select(\'li\')}</ul>\n                  <ul py:with="items=(1, 2, 3)">\n                    <xi:include href="tmpl1.html" py:for="item in items" />\n                  </ul>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  <ul><li>1</li><li>2</li><li>3</li></ul>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_fallback_when_include_found(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                    Missing</xi:fallback></xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  <div>Included</div>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_error_when_include_not_found(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl1.html"/>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname], auto_reload=True)
            tmpl = loader.load('tmpl2.html')
            self.assertRaises(TemplateNotFound, tmpl.generate().render)
        finally:
            shutil.rmtree(dirname)

    def test_fallback_when_include_not_found(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                  Missing</xi:fallback></xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  Missing\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_fallback_when_auto_reload_true(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                    Missing</xi:fallback></xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname], auto_reload=True)
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                    Missing\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_include_in_fallback(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl2.html">\n                    <xi:fallback>\n                      <xi:include href="tmpl1.html">\n                        <xi:fallback>Missing</xi:fallback>\n                      </xi:include>\n                    </xi:fallback>\n                  </xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl3.html')
            self.assertEqual('<html>\n                      <div>Included</div>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_nested_include_fallback(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file2 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl2.html">\n                    <xi:fallback>\n                      <xi:include href="tmpl1.html">\n                        <xi:fallback>Missing</xi:fallback>\n                      </xi:include>\n                    </xi:fallback>\n                  </xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl3.html')
            self.assertEqual('<html>\n                      Missing\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_nested_include_in_fallback(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file1.write('<div>Included</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude">\n                  <xi:include href="tmpl2.html">\n                    <xi:fallback>\n                      <xi:include href="tmpl1.html" />\n                    </xi:fallback>\n                  </xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl3.html')
            self.assertEqual('<html>\n                  <div>Included</div>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_include_fallback_with_directive(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                      xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="tmpl1.html"><xi:fallback>\n                    <py:if test="True">tmpl1.html not found</py:if>\n                  </xi:fallback></xi:include>\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                    tmpl1.html not found\n                </html>', tmpl.generate(debug=True).render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_include_inlined(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="tmpl1.html" />\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname], auto_reload=False)
            tmpl = loader.load('tmpl2.html')
            self.assertEqual(7, len(tmpl.stream))
            self.assertEqual('<html>\n                  <div>Included</div>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_include_inlined_in_loop(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div>Included $idx</div>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/">\n                  <xi:include href="tmpl1.html" py:for="idx in range(3)" />\n                </html>')
            finally:
                file2.close()
            loader = TemplateLoader([dirname], auto_reload=False)
            tmpl = loader.load('tmpl2.html')
            self.assertEqual('<html>\n                  <div>Included 0</div><div>Included 1</div><div>Included 2</div>\n                </html>', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_include_inline_recursive(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<div xmlns:xi="http://www.w3.org/2001/XInclude"                xmlns:py="http://genshi.edgewall.org/">$depth<py:with vars="depth = depth + 1"><xi:include href="tmpl1.html"            py:if="depth &lt; 3"/></py:with></div>')
            finally:
                file1.close()
            loader = TemplateLoader([dirname], auto_reload=False)
            tmpl = loader.load(os.path.join(dirname, 'tmpl1.html'))
            self.assertEqual('<div>0<div>1<div>2</div></div></div>', tmpl.generate(depth=0).render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_allow_exec_false(self):
        xml = '<?python\n          title = "A Genshi Template"\n          ?>\n          <html xmlns:py="http://genshi.edgewall.org/">\n            <head>\n              <title py:content="title">This is replaced.</title>\n            </head>\n        </html>'
        try:
            tmpl = MarkupTemplate(xml, filename='test.html', allow_exec=False)
            self.fail('Expected SyntaxError')
        except TemplateSyntaxError as e:
            pass

    def test_allow_exec_true(self):
        xml = '<?python\n          title = "A Genshi Template"\n          ?>\n          <html xmlns:py="http://genshi.edgewall.org/">\n            <head>\n              <title py:content="title">This is replaced.</title>\n            </head>\n        </html>'
        tmpl = MarkupTemplate(xml, filename='test.html', allow_exec=True)

    def test_exec_in_match(self):
        xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/p">\n            <?python title="wakka wakka wakka" ?>\n            ${title}\n          </py:match>\n          <body><p>moot text</p></body>\n        </html>'
        tmpl = MarkupTemplate(xml, filename='test.html', allow_exec=True)
        self.assertEqual('<html>\n          <body>\n            wakka wakka wakka\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_with_in_match(self):
        xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body/p">\n            <h1>${select(\'text()\')}</h1>\n            ${select(\'.\')}\n          </py:match>\n          <body><p py:with="foo=\'bar\'">${foo}</p></body>\n        </html>'
        tmpl = MarkupTemplate(xml, filename='test.html')
        self.assertEqual('<html>\n          <body>\n            <h1>bar</h1>\n            <p>bar</p>\n          </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_nested_include_matches(self):
        dirname = tempfile.mkdtemp(suffix='genshi_test')
        try:
            file1 = open(os.path.join(dirname, 'tmpl1.html'), 'w')
            try:
                file1.write('<html xmlns:py="http://genshi.edgewall.org/" py:strip="">\n   <div class="target">Some content.</div>\n</html>')
            finally:
                file1.close()
            file2 = open(os.path.join(dirname, 'tmpl2.html'), 'w')
            try:
                file2.write('<html xmlns:py="http://genshi.edgewall.org/"\n    xmlns:xi="http://www.w3.org/2001/XInclude">\n  <body>\n    <h1>Some full html document that includes file1.html</h1>\n    <xi:include href="tmpl1.html" />\n  </body>\n</html>')
            finally:
                file2.close()
            file3 = open(os.path.join(dirname, 'tmpl3.html'), 'w')
            try:
                file3.write('<html xmlns:py="http://genshi.edgewall.org/"\n    xmlns:xi="http://www.w3.org/2001/XInclude" py:strip="">\n  <div py:match="div[@class=\'target\']" py:attrs="select(\'@*\')">\n    Some added stuff.\n    ${select(\'*|text()\')}\n  </div>\n  <xi:include href="tmpl2.html" />\n</html>\n')
            finally:
                file3.close()
            loader = TemplateLoader([dirname])
            tmpl = loader.load('tmpl3.html')
            self.assertEqual('\n  <html>\n  <body>\n    <h1>Some full html document that includes file1.html</h1>\n   <div class="target">\n    Some added stuff.\n    Some content.\n  </div>\n  </body>\n</html>\n', tmpl.generate().render(encoding=None))
        finally:
            shutil.rmtree(dirname)

    def test_nested_matches_without_buffering(self):
        xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body" once="true" buffer="false">\n            <body>\n              ${select(\'*|text\')}\n              And some other stuff...\n            </body>\n          </py:match>\n          <body>\n            <span py:match="span">Foo</span>\n            <span>Bar</span>\n          </body>\n        </html>'
        tmpl = MarkupTemplate(xml, filename='test.html')
        self.assertEqual('<html>\n            <body>\n              <span>Foo</span>\n              And some other stuff...\n            </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_without_select(self):
        xml = '<html xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="body" buffer="false">\n            <body>\n              This replaces the other text.\n            </body>\n          </py:match>\n          <body>\n            This gets replaced.\n          </body>\n        </html>'
        tmpl = MarkupTemplate(xml, filename='test.html')
        self.assertEqual('<html>\n            <body>\n              This replaces the other text.\n            </body>\n        </html>', tmpl.generate().render(encoding=None))

    def test_match_tail_handling(self):
        xml = '<rhyme xmlns:py="http://genshi.edgewall.org/">\n          <py:match path="*[@type]">\n            ${select(\'.\')}\n          </py:match>\n\n          <lines>\n            <first type="one">fish</first>\n            <second type="two">fish</second>\n            <third type="red">fish</third>\n            <fourth type="blue">fish</fourth>\n          </lines>\n        </rhyme>'
        tmpl = MarkupTemplate(xml, filename='test.html')
        self.assertEqual('<rhyme>\n          <lines>\n            <first type="one">fish</first>\n            <second type="two">fish</second>\n            <third type="red">fish</third>\n            <fourth type="blue">fish</fourth>\n          </lines>\n        </rhyme>', tmpl.generate().render(encoding=None))

    def test_directive_single_line_with_translator(self):
        tmpl = MarkupTemplate('<div xmlns:py="http://genshi.edgewall.org/">\n            <py:for each="i in range(2)"><py:for each="j in range(1)">\n                <span py:content="i + j"></span>\n            </py:for></py:for>\n        </div>')
        translator = Translator(lambda s: s)
        tmpl.add_directives(Translator.NAMESPACE, translator)
        self.assertEqual('<div>\n                <span>0</span>\n                <span>1</span>\n        </div>', str(tmpl.generate()))