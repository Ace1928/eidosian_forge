from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_source_project_not_for_cloud(run_module, admin_user, base_inventory, project):
    result = run_module('inventory_source', dict(name='Test ec2 Inventory Source', inventory=base_inventory.name, source='ec2', state='present', source_project=project.name), admin_user)
    assert result.pop('failed', None) is True, result
    assert 'Cannot set source_project if not SCM type' in result.get('msg', '')