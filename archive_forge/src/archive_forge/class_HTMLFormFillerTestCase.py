import unittest
import six
from genshi.input import HTML, ParseError
from genshi.filters.html import HTMLFormFiller, HTMLSanitizer
from genshi.template import MarkupTemplate
from genshi.tests.test_utils import doctest_suite
class HTMLFormFillerTestCase(unittest.TestCase):

    def test_fill_input_text_no_value(self):
        html = HTML(u'<form><p>\n          <input type="text" name="foo" />\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <input type="text" name="foo"/>\n        </p></form>', html.render())

    def test_fill_input_text_single_value(self):
        html = HTML(u'<form><p>\n          <input type="text" name="foo" />\n        </p></form>') | HTMLFormFiller(data={'foo': 'bar'})
        self.assertEqual('<form><p>\n          <input type="text" name="foo" value="bar"/>\n        </p></form>', html.render())

    def test_fill_input_text_multi_value(self):
        html = HTML(u'<form><p>\n          <input type="text" name="foo" />\n        </p></form>') | HTMLFormFiller(data={'foo': ['bar']})
        self.assertEqual('<form><p>\n          <input type="text" name="foo" value="bar"/>\n        </p></form>', html.render())

    def test_fill_input_hidden_no_value(self):
        html = HTML(u'<form><p>\n          <input type="hidden" name="foo" />\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <input type="hidden" name="foo"/>\n        </p></form>', html.render())

    def test_fill_input_hidden_single_value(self):
        html = HTML(u'<form><p>\n          <input type="hidden" name="foo" />\n        </p></form>') | HTMLFormFiller(data={'foo': 'bar'})
        self.assertEqual('<form><p>\n          <input type="hidden" name="foo" value="bar"/>\n        </p></form>', html.render())

    def test_fill_input_hidden_multi_value(self):
        html = HTML(u'<form><p>\n          <input type="hidden" name="foo" />\n        </p></form>') | HTMLFormFiller(data={'foo': ['bar']})
        self.assertEqual('<form><p>\n          <input type="hidden" name="foo" value="bar"/>\n        </p></form>', html.render())

    def test_fill_textarea_no_value(self):
        html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <textarea name="foo"/>\n        </p></form>', html.render())

    def test_fill_textarea_single_value(self):
        html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n        </p></form>') | HTMLFormFiller(data={'foo': 'bar'})
        self.assertEqual('<form><p>\n          <textarea name="foo">bar</textarea>\n        </p></form>', html.render())

    def test_fill_textarea_multi_value(self):
        html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n        </p></form>') | HTMLFormFiller(data={'foo': ['bar']})
        self.assertEqual('<form><p>\n          <textarea name="foo">bar</textarea>\n        </p></form>', html.render())

    def test_fill_textarea_multiple(self):
        html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n          <textarea name="bar"></textarea>\n        </p></form>') | HTMLFormFiller(data={'foo': 'Some text'})
        self.assertEqual('<form><p>\n          <textarea name="foo">Some text</textarea>\n          <textarea name="bar"/>\n        </p></form>', html.render())

    def test_fill_textarea_preserve_original(self):
        html = HTML(u'<form><p>\n          <textarea name="foo"></textarea>\n          <textarea name="bar">Original value</textarea>\n        </p></form>') | HTMLFormFiller(data={'foo': 'Some text'})
        self.assertEqual('<form><p>\n          <textarea name="foo">Some text</textarea>\n          <textarea name="bar">Original value</textarea>\n        </p></form>', html.render())

    def test_fill_input_checkbox_single_value_auto_no_value(self):
        html = HTML(u'<form><p>\n          <input type="checkbox" name="foo" />\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo"/>\n        </p></form>', html.render())

    def test_fill_input_checkbox_single_value_auto(self):
        html = HTML(u'<form><p>\n          <input type="checkbox" name="foo" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ''})).render())
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': 'on'})).render())

    def test_fill_input_checkbox_single_value_defined(self):
        html = HTML('<form><p>\n          <input type="checkbox" name="foo" value="1" />\n        </p></form>', encoding='ascii')
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '1'})).render())
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '2'})).render())

    def test_fill_input_checkbox_multi_value_auto(self):
        html = HTML('<form><p>\n          <input type="checkbox" name="foo" />\n        </p></form>', encoding='ascii')
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': []})).render())
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['on']})).render())

    def test_fill_input_checkbox_multi_value_defined(self):
        html = HTML(u'<form><p>\n          <input type="checkbox" name="foo" value="1" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['1']})).render())
        self.assertEqual('<form><p>\n          <input type="checkbox" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['2']})).render())

    def test_fill_input_radio_no_value(self):
        html = HTML(u'<form><p>\n          <input type="radio" name="foo" />\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <input type="radio" name="foo"/>\n        </p></form>', html.render())

    def test_fill_input_radio_single_value(self):
        html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="1" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '1'})).render())
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': '2'})).render())

    def test_fill_input_radio_multi_value(self):
        html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="1" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['1']})).render())
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="1"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['2']})).render())

    def test_fill_input_radio_empty_string(self):
        html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ''})).render())

    def test_fill_input_radio_multi_empty_string(self):
        html = HTML(u'<form><p>\n          <input type="radio" name="foo" value="" />\n        </p></form>')
        self.assertEqual('<form><p>\n          <input type="radio" name="foo" value="" checked="checked"/>\n        </p></form>', (html | HTMLFormFiller(data={'foo': ['']})).render())

    def test_fill_select_no_value_auto(self):
        html = HTML(u'<form><p>\n          <select name="foo">\n            <option>1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <select name="foo">\n            <option>1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_select_no_value_defined(self):
        html = HTML(u'<form><p>\n          <select name="foo">\n            <option value="1">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n          </select>\n        </p></form>') | HTMLFormFiller()
        self.assertEqual('<form><p>\n          <select name="foo">\n            <option value="1">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_select_single_value_auto(self):
        html = HTML(u'<form><p>\n          <select name="foo">\n            <option>1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>') | HTMLFormFiller(data={'foo': '1'})
        self.assertEqual('<form><p>\n          <select name="foo">\n            <option selected="selected">1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_select_single_value_defined(self):
        html = HTML(u'<form><p>\n          <select name="foo">\n            <option value="1">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n          </select>\n        </p></form>') | HTMLFormFiller(data={'foo': '1'})
        self.assertEqual('<form><p>\n          <select name="foo">\n            <option value="1" selected="selected">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_select_multi_value_auto(self):
        html = HTML(u'<form><p>\n          <select name="foo" multiple>\n            <option>1</option>\n            <option>2</option>\n            <option>3</option>\n          </select>\n        </p></form>') | HTMLFormFiller(data={'foo': ['1', '3']})
        self.assertEqual('<form><p>\n          <select name="foo" multiple="multiple">\n            <option selected="selected">1</option>\n            <option>2</option>\n            <option selected="selected">3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_select_multi_value_defined(self):
        html = HTML(u'<form><p>\n          <select name="foo" multiple>\n            <option value="1">1</option>\n            <option value="2">2</option>\n            <option value="3">3</option>\n          </select>\n        </p></form>') | HTMLFormFiller(data={'foo': ['1', '3']})
        self.assertEqual('<form><p>\n          <select name="foo" multiple="multiple">\n            <option value="1" selected="selected">1</option>\n            <option value="2">2</option>\n            <option value="3" selected="selected">3</option>\n          </select>\n        </p></form>', html.render())

    def test_fill_option_segmented_text(self):
        html = MarkupTemplate(u'<form>\n          <select name="foo">\n            <option value="1">foo $x</option>\n          </select>\n        </form>').generate(x=1) | HTMLFormFiller(data={'foo': '1'})
        self.assertEqual(u'<form>\n          <select name="foo">\n            <option value="1" selected="selected">foo 1</option>\n          </select>\n        </form>', html.render())

    def test_fill_option_segmented_text_no_value(self):
        html = MarkupTemplate('<form>\n          <select name="foo">\n            <option>foo $x bar</option>\n          </select>\n        </form>').generate(x=1) | HTMLFormFiller(data={'foo': 'foo 1 bar'})
        self.assertEqual('<form>\n          <select name="foo">\n            <option selected="selected">foo 1 bar</option>\n          </select>\n        </form>', html.render())

    def test_fill_option_unicode_value(self):
        html = HTML(u'<form>\n          <select name="foo">\n            <option value="&ouml;">foo</option>\n          </select>\n        </form>') | HTMLFormFiller(data={'foo': u'ö'})
        self.assertEqual(u'<form>\n          <select name="foo">\n            <option value="ö" selected="selected">foo</option>\n          </select>\n        </form>', html.render(encoding=None))

    def test_fill_input_password_disabled(self):
        html = HTML(u'<form><p>\n          <input type="password" name="pass" />\n        </p></form>') | HTMLFormFiller(data={'pass': 'bar'})
        self.assertEqual('<form><p>\n          <input type="password" name="pass"/>\n        </p></form>', html.render())

    def test_fill_input_password_enabled(self):
        html = HTML(u'<form><p>\n          <input type="password" name="pass" />\n        </p></form>') | HTMLFormFiller(data={'pass': '1234'}, passwords=True)
        self.assertEqual('<form><p>\n          <input type="password" name="pass" value="1234"/>\n        </p></form>', html.render())