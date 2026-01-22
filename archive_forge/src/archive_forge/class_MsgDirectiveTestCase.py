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
class MsgDirectiveTestCase(unittest.TestCase):

    def test_extract_i18n_msg(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html">Help</a> for details.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Please see [1:Help] for details.', messages[0][2])

    def test_translate_i18n_msg(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html">Help</a> for details.\n          </p>\n        </html>')
        gettext = lambda s: u'Für Details siehe bitte [1:Hilfe].'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Für Details siehe bitte <a href="help.html">Hilfe</a>.</p>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))

    def test_extract_i18n_msg_nonewline(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">Please see <a href="help.html">Help</a></p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Please see [1:Help]', messages[0][2])

    def test_translate_i18n_msg_nonewline(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">Please see <a href="help.html">Help</a></p>\n        </html>')
        gettext = lambda s: u'Für Details siehe bitte [1:Hilfe]'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Für Details siehe bitte <a href="help.html">Hilfe</a></p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_elt_nonewline(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg>Please see <a href="help.html">Help</a></i18n:msg>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Please see [1:Help]', messages[0][2])

    def test_translate_i18n_msg_elt_nonewline(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg>Please see <a href="help.html">Help</a></i18n:msg>\n        </html>')
        gettext = lambda s: u'Für Details siehe bitte [1:Hilfe]'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          Für Details siehe bitte <a href="help.html">Hilfe</a>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))

    def test_extract_i18n_msg_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="A helpful paragraph">\n            Please see <a href="help.html" title="Click for help">Help</a>\n          </p>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(3, len(messages))
        self.assertEqual('A helpful paragraph', messages[0][2])
        self.assertEqual(3, messages[0][0])
        self.assertEqual('Click for help', messages[1][2])
        self.assertEqual(4, messages[1][0])
        self.assertEqual('Please see [1:Help]', messages[2][2])
        self.assertEqual(3, messages[2][0])

    def test_translate_i18n_msg_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="A helpful paragraph">\n            Please see <a href="help.html" title="Click for help">Help</a>\n          </p>\n        </html>')
        translator = Translator(lambda msgid: {'A helpful paragraph': 'Ein hilfreicher Absatz', 'Click for help': u'Klicken für Hilfe', 'Please see [1:Help]': u'Siehe bitte [1:Hilfe]'}[msgid])
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p title="Ein hilfreicher Absatz">Siehe bitte <a href="help.html" title="Klicken für Hilfe">Hilfe</a></p>\n        </html>', tmpl.generate().render(encoding=None))

    def test_extract_i18n_msg_with_dynamic_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="${_(\'A helpful paragraph\')}">\n            Please see <a href="help.html" title="${_(\'Click for help\')}">Help</a>\n          </p>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(3, len(messages))
        self.assertEqual('A helpful paragraph', messages[0][2])
        self.assertEqual(3, messages[0][0])
        self.assertEqual('Click for help', messages[1][2])
        self.assertEqual(4, messages[1][0])
        self.assertEqual('Please see [1:Help]', messages[2][2])
        self.assertEqual(3, messages[2][0])

    def test_translate_i18n_msg_with_dynamic_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="${_(\'A helpful paragraph\')}">\n            Please see <a href="help.html" title="${_(\'Click for help\')}">Help</a>\n          </p>\n        </html>')
        translator = Translator(lambda msgid: {'A helpful paragraph': 'Ein hilfreicher Absatz', 'Click for help': u'Klicken für Hilfe', 'Please see [1:Help]': u'Siehe bitte [1:Hilfe]'}[msgid])
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p title="Ein hilfreicher Absatz">Siehe bitte <a href="help.html" title="Klicken für Hilfe">Hilfe</a></p>\n        </html>', tmpl.generate(_=translator.translate).render(encoding=None))

    def test_extract_i18n_msg_as_element_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="">\n            Please see <a href="help.html" title="Click for help">Help</a>\n          </i18n:msg>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual('Click for help', messages[0][2])
        self.assertEqual(4, messages[0][0])
        self.assertEqual('Please see [1:Help]', messages[1][2])
        self.assertEqual(3, messages[1][0])

    def test_translate_i18n_msg_as_element_with_attributes(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="">\n            Please see <a href="help.html" title="Click for help">Help</a>\n          </i18n:msg>\n        </html>')
        translator = Translator(lambda msgid: {'Click for help': u'Klicken für Hilfe', 'Please see [1:Help]': u'Siehe bitte [1:Hilfe]'}[msgid])
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          Siehe bitte <a href="help.html" title="Klicken für Hilfe">Hilfe</a>\n        </html>', tmpl.generate().render(encoding=None))

    def test_extract_i18n_msg_nested(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html"><em>Help</em> page</a> for details.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Please see [1:[2:Help] page] for details.', messages[0][2])

    def test_translate_i18n_msg_nested(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html"><em>Help</em> page</a> for details.\n          </p>\n        </html>')
        gettext = lambda s: u'Für Details siehe bitte [1:[2:Hilfeseite]].'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Für Details siehe bitte <a href="help.html"><em>Hilfeseite</em></a>.</p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_label_with_nested_input(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:msg="">\n            <label><input type="text" size="3" name="daysback" value="30" /> days back</label>\n          </div>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('[1:[2:] days back]', messages[0][2])

    def test_translate_i18n_msg_label_with_nested_input(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <div i18n:msg="">\n            <label><input type="text" size="3" name="daysback" value="30" /> foo bar</label>\n          </div>\n        </html>')
        gettext = lambda s: '[1:[2:] foo bar]'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <div><label><input type="text" size="3" name="daysback" value="30"/> foo bar</label></div>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_empty(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" /> entries per page.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Show me [1:] entries per page.', messages[0][2])

    def test_translate_i18n_msg_empty(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" /> entries per page.\n          </p>\n        </html>')
        gettext = lambda s: u'[1:] Einträge pro Seite anzeigen.'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p><input type="text" name="num"/> Einträge pro Seite anzeigen.</p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_multiple(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html">Help</a> for <em>details</em>.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Please see [1:Help] for [2:details].', messages[0][2])

    def test_translate_i18n_msg_multiple(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Please see <a href="help.html">Help</a> for <em>details</em>.\n          </p>\n        </html>')
        gettext = lambda s: u'Für [2:Details] siehe bitte [1:Hilfe].'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Für <em>Details</em> siehe bitte <a href="help.html">Hilfe</a>.</p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_multiple_empty(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" /> entries per page, starting at page <input type="text" name="num" />.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Show me [1:] entries per page, starting at page [2:].', messages[0][2])

    def test_translate_i18n_msg_multiple_empty(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" /> entries per page, starting at page <input type="text" name="num" />.\n          </p>\n        </html>', encoding='utf-8')
        gettext = lambda s: u'[1:] Einträge pro Seite, beginnend auf Seite [2:].'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p><input type="text" name="num"/> Einträge pro Seite, beginnend auf Seite <input type="text" name="num"/>.</p>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))

    def test_extract_i18n_msg_with_param(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name">\n            Hello, ${user.name}!\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Hello, %(name)s!', messages[0][2])

    def test_translate_i18n_msg_with_param(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name">\n            Hello, ${user.name}!\n          </p>\n        </html>')
        gettext = lambda s: u'Hallo, %(name)s!'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Hallo, Jim!</p>\n        </html>', tmpl.generate(user=dict(name='Jim')).render())

    def test_translate_i18n_msg_with_param_reordered(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name">\n            Hello, ${user.name}!\n          </p>\n        </html>')
        gettext = lambda s: u'%(name)s, sei gegrüßt!'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Jim, sei gegrüßt!</p>\n        </html>', tmpl.generate(user=dict(name='Jim')).render())

    def test_translate_i18n_msg_with_attribute_param(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Hello, <a href="#${anchor}">dude</a>!\n          </p>\n        </html>')
        gettext = lambda s: u'Sei gegrüßt, [1:Alter]!'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p>Sei gegrüßt, <a href="#42">Alter</a>!</p>\n        </html>', tmpl.generate(anchor='42').render())

    def test_extract_i18n_msg_with_two_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name, time">\n            Posted by ${post.author} at ${entry.time.strftime(\'%H:%m\')}\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Posted by %(name)s at %(time)s', messages[0][2])

    def test_translate_i18n_msg_with_two_params(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="name, time">\n            Written by ${entry.author} at ${entry.time.strftime(\'%H:%M\')}\n          </p>\n        </html>')
        gettext = lambda s: u'%(name)s schrieb dies um %(time)s'
        translator = Translator(gettext)
        translator.setup(tmpl)
        entry = {'author': 'Jim', 'time': datetime(2008, 4, 1, 14, 30)}
        self.assertEqual('<html>\n          <p>Jim schrieb dies um 14:30</p>\n        </html>', tmpl.generate(entry=entry).render())

    def test_extract_i18n_msg_with_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" py:attrs="{\'value\': x}" /> entries per page.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Show me [1:] entries per page.', messages[0][2])

    def test_translate_i18n_msg_with_directive(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">\n            Show me <input type="text" name="num" py:attrs="{\'value\': \'x\'}" /> entries per page.\n          </p>\n        </html>')
        gettext = lambda s: u'[1:] Einträge pro Seite anzeigen.'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p><input type="text" name="num" value="x"/> Einträge pro Seite anzeigen.</p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_with_comment(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:comment="As in foo bar" i18n:msg="">Foo</p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, 'Foo', ['As in foo bar']), messages[0])
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" i18n:comment="As in foo bar">Foo</p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, 'Foo', ['As in foo bar']), messages[0])

    def test_translate_i18n_msg_with_comment(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" i18n:comment="As in foo bar">Foo</p>\n        </html>')
        gettext = lambda s: u'Voh'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>Voh</p>\n        </html>', tmpl.generate().render())

    def test_extract_i18n_msg_with_attr(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="Foo bar">Foo</p>\n        </html>')
        translator = Translator()
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(2, len(messages))
        self.assertEqual((3, None, 'Foo bar', []), messages[0])
        self.assertEqual((3, None, 'Foo', []), messages[1])

    def test_translate_i18n_msg_with_attr(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" title="Foo bar">Foo</p>\n        </html>')
        gettext = lambda s: u'Voh'
        translator = Translator(DummyTranslations({'Foo': 'Voh', 'Foo bar': u'Voh bär'}))
        tmpl.filters.insert(0, translator)
        tmpl.add_directives(Translator.NAMESPACE, translator)
        self.assertEqual(u'<html>\n          <p title="Voh bär">Voh</p>\n        </html>', tmpl.generate().render())

    def test_translate_i18n_msg_and_py_strip_directives(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" py:strip="">Foo</p>\n          <p py:strip="" i18n:msg="">Foo</p>\n        </html>')
        translator = Translator(DummyTranslations({'Foo': 'Voh'}))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          Voh\n          Voh\n        </html>', tmpl.generate().render())

    def test_i18n_msg_ticket_300_extract(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="date, author">\n            Changed ${ \'10/12/2008\' } ago by ${ \'me, the author\' }\n          </i18n:msg>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, 'Changed %(date)s ago by %(author)s', []), messages[0])

    def test_i18n_msg_ticket_300_translate(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <i18n:msg params="date, author">\n            Changed ${ date } ago by ${ author }\n          </i18n:msg>\n        </html>')
        translations = DummyTranslations({'Changed %(date)s ago by %(author)s': u'Modificado à %(date)s por %(author)s'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          Modificado à um dia por Pedro\n        </html>'.encode('utf-8'), tmpl.generate(date='um dia', author='Pedro').render(encoding='utf-8'))

    def test_i18n_msg_ticket_251_extract(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg=""><tt><b>Translation[&nbsp;0&nbsp;]</b>: <em>One coin</em></tt></p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, u'[1:[2:Translation\\[\xa00\xa0\\]]: [3:One coin]]', []), messages[0])

    def test_i18n_msg_ticket_251_translate(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg=""><tt><b>Translation[&nbsp;0&nbsp;]</b>: <em>One coin</em></tt></p>\n        </html>')
        translations = DummyTranslations({u'[1:[2:Translation\\[\xa00\xa0\\]]: [3:One coin]]': u'[1:[2:Trandução\\[\xa00\xa0\\]]: [3:Uma moeda]]'})
        translator = Translator(translations)
        translator.setup(tmpl)
        self.assertEqual(u'<html>\n          <p><tt><b>Trandução[\xa00\xa0]</b>: <em>Uma moeda</em></tt></p>\n        </html>'.encode('utf-8'), tmpl.generate().render(encoding='utf-8'))

    def test_extract_i18n_msg_with_other_directives_nested(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" py:with="q = quote_plus(message[:80])">Before you do that, though, please first try\n            <strong><a href="${trac.homepage}search?ticket=yes&amp;noquickjump=1&amp;q=$q">searching</a>\n            for similar issues</strong>, as it is quite likely that this problem\n            has been reported before. For questions about installation\n            and configuration of Trac, please try the\n            <a href="${trac.homepage}wiki/MailingList">mailing list</a>\n            instead of filing a ticket.\n          </p>\n        </html>')
        translator = Translator()
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('Before you do that, though, please first try\n            [1:[2:searching]\n            for similar issues], as it is quite likely that this problem\n            has been reported before. For questions about installation\n            and configuration of Trac, please try the\n            [3:mailing list]\n            instead of filing a ticket.', messages[0][2])

    def test_translate_i18n_msg_with_other_directives_nested(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="">Before you do that, though, please first try\n            <strong><a href="${trac.homepage}search?ticket=yes&amp;noquickjump=1&amp;q=q">searching</a>\n            for similar issues</strong>, as it is quite likely that this problem\n            has been reported before. For questions about installation\n            and configuration of Trac, please try the\n            <a href="${trac.homepage}wiki/MailingList">mailing list</a>\n            instead of filing a ticket.\n          </p>\n        </html>')
        translations = DummyTranslations({'Before you do that, though, please first try\n            [1:[2:searching]\n            for similar issues], as it is quite likely that this problem\n            has been reported before. For questions about installation\n            and configuration of Trac, please try the\n            [3:mailing list]\n            instead of filing a ticket.': u'Antes de o fazer, porém,\n            [1:por favor tente [2:procurar]\n            por problemas semelhantes], uma vez que é muito provável que este problema\n            já tenha sido reportado anteriormente. Para questões relativas à instalação\n            e configuração do Trac, por favor tente a\n            [3:mailing list]\n            em vez de criar um assunto.'})
        translator = Translator(translations)
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        ctx = Context()
        ctx.push({'trac': {'homepage': 'http://trac.edgewall.org/'}})
        self.assertEqual(u'<html>\n          <p>Antes de o fazer, porém,\n            <strong>por favor tente <a href="http://trac.edgewall.org/search?ticket=yes&amp;noquickjump=1&amp;q=q">procurar</a>\n            por problemas semelhantes</strong>, uma vez que é muito provável que este problema\n            já tenha sido reportado anteriormente. Para questões relativas à instalação\n            e configuração do Trac, por favor tente a\n            <a href="http://trac.edgewall.org/wiki/MailingList">mailing list</a>\n            em vez de criar um assunto.</p>\n        </html>', tmpl.generate(ctx).render())

    def test_i18n_msg_with_other_nested_directives_with_reordered_content(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p py:if="not editable" class="hint" i18n:msg="">\n            <strong>Note:</strong> This repository is defined in\n            <code><a href="${ \'href.wiki(TracIni)\' }">trac.ini</a></code>\n            and cannot be edited on this page.\n          </p>\n        </html>')
        translations = DummyTranslations({'[1:Note:] This repository is defined in\n            [2:[3:trac.ini]]\n            and cannot be edited on this page.': u'[1:Nota:] Este repositório está definido em \n           [2:[3:trac.ini]]\n            e não pode ser editado nesta página.'})
        translator = Translator(translations)
        translator.setup(tmpl)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual('[1:Note:] This repository is defined in\n            [2:[3:trac.ini]]\n            and cannot be edited on this page.', messages[0][2])
        self.assertEqual(u'<html>\n          <p class="hint"><strong>Nota:</strong> Este repositório está definido em\n           <code><a href="href.wiki(TracIni)">trac.ini</a></code>\n            e não pode ser editado nesta página.</p>\n        </html>'.encode('utf-8'), tmpl.generate(editable=False).render(encoding='utf-8'))

    def test_extract_i18n_msg_with_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" py:strip="">\n            Please see <a href="help.html">Help</a> for details.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, 'Please see [1:Help] for details.', []), messages[0])

    def test_extract_i18n_msg_with_py_strip_and_comment(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" py:strip="" i18n:comment="Foo">\n            Please see <a href="help.html">Help</a> for details.\n          </p>\n        </html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(1, len(messages))
        self.assertEqual((3, None, 'Please see [1:Help] for details.', ['Foo']), messages[0])

    def test_translate_i18n_msg_and_comment_with_py_strip_directives(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="" i18n:comment="As in foo bar" py:strip="">Foo</p>\n          <p py:strip="" i18n:msg="" i18n:comment="As in foo bar">Foo</p>\n        </html>')
        translator = Translator(DummyTranslations({'Foo': 'Voh'}))
        translator.setup(tmpl)
        self.assertEqual('<html>\n          Voh\n          Voh\n        </html>', tmpl.generate().render())

    def test_translate_i18n_msg_ticket_404(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <p i18n:msg="first,second">\n            $first <span>$second</span> KEPT <span>Inside a tag</span> tail\n          </p></html>')
        translator = Translator(DummyTranslations())
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <p>FIRST <span>SECOND</span> KEPT <span>Inside a tag</span> tail</p></html>', tmpl.generate(first='FIRST', second='SECOND').render())

    def test_translate_i18n_msg_ticket_404_regression(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/"\n            xmlns:i18n="http://genshi.edgewall.org/i18n">\n          <h1 i18n:msg="name">text <a>$name</a></h1>\n        </html>')
        gettext = lambda s: u'head [1:%(name)s] tail'
        translator = Translator(gettext)
        translator.setup(tmpl)
        self.assertEqual('<html>\n          <h1>head <a>NAME</a> tail</h1>\n        </html>', tmpl.generate(name='NAME').render())