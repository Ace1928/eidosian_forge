from unittest import mock
from oslotest import base as test_base
from oslo_i18n import _factory
from oslo_i18n import _lazy
from oslo_i18n import _message
def test_not_lazy(self):
    _lazy.enable_lazy(False)
    with mock.patch.object(_message, 'Message') as msg:
        msg.side_effect = AssertionError('should not use Message')
        tf = _factory.TranslatorFactory('domain')
        tf.primary('some text')