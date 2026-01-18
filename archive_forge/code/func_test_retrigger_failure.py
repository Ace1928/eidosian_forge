import copy
import json
import time
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
@test.requires_convergence
def test_retrigger_failure(self):
    before, after = get_templates(fail=True)
    stack_id = self.stack_create(template=before, expected_status='CREATE_IN_PROGRESS')
    time.sleep(10)
    self.update_stack(stack_id, after)