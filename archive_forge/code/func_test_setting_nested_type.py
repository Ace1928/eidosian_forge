from __future__ import absolute_import, division, print_function
import pytest
from awx.conf.models import Setting
@pytest.mark.django_db
def test_setting_nested_type(run_module, admin_user):
    the_value = {'email': 'mail', 'first_name': 'givenName', 'last_name': 'surname'}
    result = run_module('settings', dict(settings={'AUTH_LDAP_USER_ATTR_MAP': the_value}), admin_user)
    assert not result.get('failed', False), result.get('msg', result)
    assert result.get('changed'), result
    assert Setting.objects.get(key='AUTH_LDAP_USER_ATTR_MAP').value == the_value