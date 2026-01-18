import fixtures
from glance.common import exception
from glance.common import swift_store_utils
from glance.tests.unit import base
def test_swift_config_uses_default_values_multiple_account_disabled(self):
    default_user = 'user_default'
    default_key = 'key_default'
    default_auth_address = 'auth@default.com'
    default_account_reference = 'ref_default'
    confs = {'swift_store_config_file': None, 'swift_store_user': default_user, 'swift_store_key': default_key, 'swift_store_auth_address': default_auth_address, 'default_swift_reference': default_account_reference}
    self.config(**confs)
    swift_params = swift_store_utils.SwiftParams().params
    self.assertEqual(1, len(swift_params.keys()))
    self.assertEqual(default_user, swift_params[default_account_reference]['user'])
    self.assertEqual(default_key, swift_params[default_account_reference]['key'])
    self.assertEqual(default_auth_address, swift_params[default_account_reference]['auth_address'])