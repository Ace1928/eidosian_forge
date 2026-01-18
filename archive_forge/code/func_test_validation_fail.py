import copy
import json
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validation_fail(self):
    param1 = {'one': 'croissant'}
    env = environment.Environment({'parameters': {'param1': json.dumps(param1)}})
    self.assertRaises(exception.StackValidationFailed, self.create_stack, self.template_bad, env)