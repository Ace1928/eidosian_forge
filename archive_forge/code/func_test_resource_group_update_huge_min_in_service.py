import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_update_huge_min_in_service(self):
    """Test rolling update with huge minimum capacity.

        Rolling Update with a huge number of minimum instances
        in service.
        """
    updt_template = yaml.safe_load(copy.deepcopy(self.template))
    grp = updt_template['resources']['random_group']
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '20'
    policy['max_batch_size'] = '1'
    res_def = grp['properties']['resource_def']
    res_def['properties']['value'] = 'updated'
    self.update_resource_group(updt_template, updated=10, created=0, deleted=0)