from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Inventory
@pytest.mark.django_db
def test_valid_smart_inventory_create(run_module, admin_user, organization):
    result = run_module('inventory', {'name': 'foo-inventory', 'organization': organization.name, 'kind': 'smart', 'host_filter': 'name=my_host', 'state': 'present'}, admin_user)
    assert not result.get('failed', False), result
    inv = Inventory.objects.get(name='foo-inventory')
    assert inv.host_filter == 'name=my_host'
    assert inv.kind == 'smart'
    assert inv.organization_id == organization.id