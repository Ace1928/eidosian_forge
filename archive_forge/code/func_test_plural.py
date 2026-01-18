import logging
from unittest import mock
import warnings
from oslotest import base as test_base
import testtools
from oslo_i18n import _message
from oslo_i18n.tests import fakes
from oslo_i18n.tests import utils
@mock.patch('gettext.translation')
def test_plural(self, translation):
    lang = mock.Mock()
    translation.return_value = lang
    trans = mock.Mock()
    trans.return_value = 'translated'
    lang.ngettext = trans
    lang.ungettext = trans
    result = _message.Message._translate_msgid(('single', 'plural', -1), domain='domain', has_contextual_form=False, has_plural_form=True)
    self.assertEqual('translated', result)
    trans.assert_called_with('single', 'plural', -1)