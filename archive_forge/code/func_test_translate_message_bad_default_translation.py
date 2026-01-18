import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
@mock.patch('locale.getlocale', return_value=('es', ''))
@mock.patch('oslo_i18n._message.LOG')
def test_translate_message_bad_default_translation(self, mock_log, mock_locale, mock_translation):
    message_with_params = 'A message: %s'
    es_translation = 'A message in Spanish: %s %s'
    param = 'A Message param'
    translations = {message_with_params: es_translation}
    translator = fakes.FakeTranslations.translator({'es': translations})
    mock_translation.side_effect = translator
    msg = _message.Message(message_with_params)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        msg = msg % param
        self.assertEqual(1, len(w))
        self.assertEqual("Failed to insert replacement values into translated message A message in Spanish: %s %s (Original: 'A message: %s'): not enough arguments for format string", str(w[0].message).replace("u'", "'"))
    mock_log.debug.assert_called_with('Failed to insert replacement values into translated message %s (Original: %r): %s', es_translation, message_with_params, mock.ANY)
    mock_log.reset_mock()
    default_translation = message_with_params % param
    self.assertEqual(default_translation, msg)
    self.assertFalse(mock_log.warning.called)