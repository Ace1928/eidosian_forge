import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def test_resource_group_update_replace_template_changed(self):
    """Test rolling update(replace)with child template path changed.

        Simple rolling update replace with child template path changed.
        """
    nested_templ = '\nheat_template_version: "2013-05-23"\nresources:\n  oops:\n    type: OS::Heat::TestResource\n'
    create_template = yaml.safe_load(copy.deepcopy(self.template))
    grp = create_template['resources']['random_group']
    grp['properties']['resource_def'] = {'type': '/opt/provider.yaml'}
    files = {'/opt/provider.yaml': nested_templ}
    policy = grp['update_policy']['rolling_update']
    policy['min_in_service'] = '1'
    policy['max_batch_size'] = '3'
    stack_identifier = self.stack_create(template=create_template, files=files)
    update_template = create_template.copy()
    grp = update_template['resources']['random_group']
    grp['properties']['resource_def'] = {'type': '/opt1/provider.yaml'}
    files = {'/opt1/provider.yaml': nested_templ}
    self.update_stack(stack_identifier, update_template, files=files)