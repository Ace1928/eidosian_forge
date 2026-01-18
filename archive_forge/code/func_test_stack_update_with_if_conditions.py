import copy
import json
from heat_integrationtests.common import test
from heat_integrationtests.functional import functional_base
def test_stack_update_with_if_conditions(self):
    test3 = {'type': 'OS::Heat::TestResource', 'properties': {'value': {'if': ['cond1', 'val3', 'val4']}}}
    self._test_conditional(test3)