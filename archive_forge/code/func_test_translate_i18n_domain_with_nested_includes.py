from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
def test_translate_i18n_domain_with_nested_includes(self):
    import os, shutil, tempfile
    from genshi.template.loader import TemplateLoader
    dirname = tempfile.mkdtemp(suffix='genshi_test')
    try:
        for idx in range(7):
            file1 = open(os.path.join(dirname, 'tmpl%d.html' % idx), 'w')
            try:
                file1.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                         xmlns:py="http://genshi.edgewall.org/"\n                                         xmlns:i18n="http://genshi.edgewall.org/i18n" py:strip="">\n                        <div>Included tmpl$idx</div>\n                        <p i18n:msg="idx">Bar $idx</p>\n                        <p i18n:domain="bar">Bar</p>\n                        <p i18n:msg="idx" i18n:domain="">Bar $idx</p>\n                        <p i18n:domain="" i18n:msg="idx">Bar $idx</p>\n                        <py:if test="idx &lt; 6">\n                        <xi:include href="tmpl${idx}.html" py:with="idx = idx+1"/>\n                        </py:if>\n                    </html>')
            finally:
                file1.close()
        file2 = open(os.path.join(dirname, 'tmpl10.html'), 'w')
        try:
            file2.write('<html xmlns:xi="http://www.w3.org/2001/XInclude"\n                                     xmlns:py="http://genshi.edgewall.org/"\n                                     xmlns:i18n="http://genshi.edgewall.org/i18n"\n                                     i18n:domain="foo">\n                  <xi:include href="tmpl${idx}.html" py:with="idx = idx+1"/>\n                </html>')
        finally:
            file2.close()

        def callback(template):
            translations = DummyTranslations({'Bar %(idx)s': 'Voh %(idx)s'})
            translations.add_domain('foo', {'Bar %(idx)s': 'foo_Bar %(idx)s'})
            translations.add_domain('bar', {'Bar': 'bar_Bar'})
            translator = Translator(translations)
            translator.setup(template)
        loader = TemplateLoader([dirname], callback=callback)
        tmpl = loader.load('tmpl10.html')
        self.assertEqual('<html>\n                        <div>Included tmpl0</div>\n                        <p>foo_Bar 0</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 0</p>\n                        <p>Voh 0</p>\n                        <div>Included tmpl1</div>\n                        <p>foo_Bar 1</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 1</p>\n                        <p>Voh 1</p>\n                        <div>Included tmpl2</div>\n                        <p>foo_Bar 2</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 2</p>\n                        <p>Voh 2</p>\n                        <div>Included tmpl3</div>\n                        <p>foo_Bar 3</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 3</p>\n                        <p>Voh 3</p>\n                        <div>Included tmpl4</div>\n                        <p>foo_Bar 4</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 4</p>\n                        <p>Voh 4</p>\n                        <div>Included tmpl5</div>\n                        <p>foo_Bar 5</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 5</p>\n                        <p>Voh 5</p>\n                        <div>Included tmpl6</div>\n                        <p>foo_Bar 6</p>\n                        <p>bar_Bar</p>\n                        <p>Voh 6</p>\n                        <p>Voh 6</p>\n                </html>', tmpl.generate(idx=-1).render())
    finally:
        shutil.rmtree(dirname)