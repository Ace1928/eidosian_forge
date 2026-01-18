import json
from unittest import mock
import betamax
from requests import models
import testtools
from keystoneauth1.fixture import hooks
@mock.patch('keystoneauth1.fixture.hooks.mask_fixture_values')
def test_pre_record_hook_empty_body(self, mask_fixture_values):
    interaction = mock.Mock()
    interaction.data = {'request': {'body': {'encoding': 'utf-8', 'string': ''}}, 'response': {'body': {'encoding': 'utf-8', 'string': ''}}}
    hooks.pre_record_hook(interaction, mock.Mock())
    self.assertFalse(mask_fixture_values.called)