from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models import Organization, Inventory, InventorySource, Project
@pytest.mark.django_db
def test_scm_source_needs_project(run_module, admin_user, base_inventory):
    result = run_module('inventory_source', dict(name='SCM inventory without project', inventory=base_inventory.name, state='present', source='scm', source_path='/var/lib/awx/example_source_path/'), admin_user)
    assert result.pop('failed', None), result
    assert 'Project required for scm type sources' in result.get('msg', '')