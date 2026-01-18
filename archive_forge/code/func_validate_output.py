import copy
import json
from heatclient import exc
import yaml
from heat_integrationtests.functional import functional_base
def validate_output(stack, output_key, length):
    output_value = self._stack_output(stack, output_key)
    self.assertEqual(length, len(output_value))
    return output_value