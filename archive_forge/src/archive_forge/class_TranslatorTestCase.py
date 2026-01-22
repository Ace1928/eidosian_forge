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
class TranslatorTestCase(unittest.TestCase):

    def test_translate_included_attribute_text(self):
        """
        Verify that translated attributes end up in a proper `Attrs` instance.
        """
        html = HTML(u'<html>\n          <span title="Foo"></span>\n        </html>')
        translator = Translator(lambda s: u'Voh')
        stream = list(html.filter(translator))
        kind, data, pos = stream[2]
        assert isinstance(data[1], Attrs)

    def test_extract_included_empty_attribute_text(self):
        tmpl = MarkupTemplate(u'<html>\n          <span title="">...</span>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual([], messages)

    def test_translate_included_empty_attribute_text(self):
        tmpl = MarkupTemplate(u'<html>\n          <span title="">...</span>\n        </html>')
        translator = Translator(DummyTranslations({'': 'Project-Id-Version'}))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <span title="">...</span>\n        </html>', tmpl.generate().render())

    def test_extract_without_text(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p title="Bar">Foo</p>\n          ${ngettext("Singular", "Plural", num)}\n        </html>')
        translator = Translator(extract_text=False)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, 'ngettext', ('Singular', 'Plural', None), []), messages[0])

    def test_extract_plural_form(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          ${ngettext("Singular", "Plural", num)}\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, 'ngettext', ('Singular', 'Plural', None), []), messages[0])

    def test_extract_funky_plural_form(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          ${ngettext(len(items), *widget.display_names)}\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, 'ngettext', (None, None), []), messages[0])

    def test_extract_gettext_with_unicode_string(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          ${gettext("Grüße")}\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, 'gettext', u'Grüße', []), messages[0])

    def test_extract_included_attribute_text(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <span title="Foo"></span>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, None, 'Foo', []), messages[0])

    def test_extract_attribute_expr(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <input type="submit" value="${_(\'Save\')}" />\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, '_', 'Save', []), messages[0])

    def test_extract_non_included_attribute_interpolated(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <a href="#anchor_${num}">Foo</a>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, None, 'Foo', []), messages[0])

    def test_extract_text_from_sub(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <py:if test="foo">Foo</py:if>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, None, 'Foo', []), messages[0])

    def test_ignore_tag_with_fixed_xml_lang(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p xml:lang="en">(c) 2007 Edgewall Software</p>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(0, len(messages))

    def test_extract_tag_with_variable_xml_lang(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <p xml:lang="${lang}">(c) 2007 Edgewall Software</p>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, None, '(c) 2007 Edgewall Software', []), messages[0])

    def test_ignore_attribute_with_expression(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <input type="submit" value="Reply" title="Reply to comment $num" />\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(0, len(messages))

    def test_translate_with_translations_object(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" i18n:comment="As in foo bar">Foo</p>\n        </html>')
        translator = Translator(DummyTranslations({'Foo': 'Voh'}))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Voh</p>\n        </html>', tmpl.generate().render())

    def test_extract_included_attribute_text_with_spaces(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <span title=" Foo ">...</span>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((2, None, 'Foo', []), messages[0])

    def test_translate_included_attribute_text_with_spaces(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/">\n          <span title=" Foo ">...</span>\n        </html>')
        translator = Translator(DummyTranslations({'Foo': 'Voh'}))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <span title="Voh">...</span>\n        </html>', tmpl.generate().render())