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