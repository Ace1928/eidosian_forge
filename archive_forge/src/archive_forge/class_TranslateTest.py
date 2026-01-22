from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _message
from oslo_i18n import _translate
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
class TranslateTest(test_base.BaseTestCase):

    @mock.patch('gettext.translation')
    def test_translate(self, mock_translation):
        en_message = 'A message in the default locale'
        es_translation = 'A message in Spanish'
        message = _message.Message(en_message)
        es_translations = {en_message: es_translation}
        translations_map = {'es': es_translations}
        translator = fakes.FakeTranslations.translator(translations_map)
        mock_translation.side_effect = translator
        obj = utils.SomeObject(message)
        self.assertEqual(es_translation, _translate.translate(message, 'es'))
        self.assertEqual(es_translation, _translate.translate(obj, 'es'))