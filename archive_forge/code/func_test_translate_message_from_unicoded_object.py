import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_translate_message_from_unicoded_object(self, mock_translation):
    en_message = 'A message in the default locale'
    es_translation = 'A message in Spanish'
    message = _message.Message(en_message)
    es_translations = {en_message: es_translation}
    translations_map = {'es': es_translations}
    translator = fakes.FakeTranslations.translator(translations_map)
    mock_translation.side_effect = translator
    obj = utils.SomeObject(message)
    unicoded_obj = str(obj)
    self.assertEqual(es_translation, unicoded_obj.translation('es'))