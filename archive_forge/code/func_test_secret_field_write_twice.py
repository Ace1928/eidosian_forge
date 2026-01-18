from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Credential, CredentialType, Organization
@pytest.mark.django_db
@pytest.mark.parametrize('update_secrets', [True, False])
def test_secret_field_write_twice(run_module, organization, admin_user, cred_type, update_secrets):
    val1 = '7rEZK38DJl58A7RxA6EC7lLvUHbBQ1'
    val2 = '7rEZ238DJl5837rxA6xxxlLvUHbBQ1'
    for val in (val1, val2):
        result = run_module('credential', dict(name='Galaxy Token for Steve', organization=organization.name, credential_type=cred_type.name, inputs={'token': val}, update_secrets=update_secrets), admin_user)
        assert not result.get('failed', False), result.get('msg', result)
        if update_secrets:
            assert Credential.objects.get(id=result['id']).get_input('token') == val
    if update_secrets:
        assert result.get('changed'), result
    else:
        assert result.get('changed') is False, result
        assert Credential.objects.get(id=result['id']).get_input('token') == val1