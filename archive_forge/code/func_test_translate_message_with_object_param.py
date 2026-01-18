import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_translate_message_with_object_param(self, mock_translation):
    message_with_params = 'A message: %s'
    es_translation = 'A message in Spanish: %s'
    param = 'A Message param'
    param_translation = 'A Message param in Spanish'
    translations = {message_with_params: es_translation, param: param_translation}
    translator = fakes.FakeTranslations.translator({'es': translations})
    mock_translation.side_effect = translator
    msg = _message.Message(message_with_params)
    param_msg = _message.Message(param)
    obj = utils.SomeObject(param_msg)
    msg = msg % obj
    default_translation = message_with_params % param
    expected_translation = es_translation % param_translation
    self.assertEqual(expected_translation, msg.translation('es'))
    self.assertEqual(default_translation, msg.translation('XX'))