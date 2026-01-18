from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def update_mandatory_parameters(self, template_params):
    """
        Update parameters which are mandatory for creating a template.

        Parameters:
            template_params (dict) - Template information.

        Returns:
            None
        """
    template_params['projectId'] = self.have_project.get('id')
    template_params['project_id'] = self.have_project.get('id')
    if not template_params.get('language'):
        template_params['language'] = self.have_template.get('template').get('language')
    if not template_params.get('deviceTypes'):
        template_params['deviceTypes'] = self.have_template.get('template').get('deviceTypes')
    if not template_params.get('softwareType'):
        template_params['softwareType'] = self.have_template.get('template').get('softwareType')