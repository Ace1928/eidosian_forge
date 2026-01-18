import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_update_scaleup(self):
    """Test rolling update with scaleup.

        Simple rolling update with increased size.
        """
    updt_template = yaml.safe_load(copy.deepcopy(self.template))
    grp = updt_template['resources']['random_group']
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '1'
    policy['max_batch_size'] = '3'
    grp['properties']['count'] = 12
    res_def = grp['properties']['resource_def']
    res_def['properties']['value'] = 'updated'
    self.update_resource_group(updt_template, updated=10, created=2, deleted=0)