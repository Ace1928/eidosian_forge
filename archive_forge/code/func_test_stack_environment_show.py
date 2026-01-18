import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
def test_stack_environment_show(self):
    columns, outputs = self._test_stack_environment_show(self.SAMPLE_ENV)
    self.assertEqual([{'p1': 'v1'}, {'resources': {'r1': 't1'}}, {'p1': 'v_default'}], outputs)