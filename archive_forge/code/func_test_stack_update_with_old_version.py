import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_old_version(self):
    """Update handles old template version in failure.

        If a stack update fails while changing the template version, update is
        able to handle the old version fine.
        """
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'value': {'list_join': [',', ['a'], ['b']]}})
    template['heat_template_version'] = '2015-10-15'
    stack_identifier = self.stack_create(template=template)
    template = _change_rsrc_properties(test_template_one_resource, ['test1'], {'fail': True})
    self.update_stack(stack_identifier, template=template, expected_status='UPDATE_FAILED')
    self._stack_delete(stack_identifier)