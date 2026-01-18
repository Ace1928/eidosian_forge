import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_update_replace(self):
    """Test rolling update(replace)with no conflict.

        Simple rolling update replace with no conflict in batch size
        and minimum instances in service.
        """
    updt_template = yaml.safe_load(copy.deepcopy(self.template))
    grp = updt_template['resources']['random_group']
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '1'
    policy['max_batch_size'] = '3'
    res_def = grp['properties']['resource_def']
    res_def['properties']['value'] = 'updated'
    res_def['properties']['update_replace'] = True
    self.update_resource_group(updt_template, updated=0, created=10, deleted=10)