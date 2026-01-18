import contextlib
from unittest import mock
from heat.common import exception as exc
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_properties_schema(self):
    if self.err:
        err = self.assertRaises(self.err, self.stack.validate)
        if self.err_msg:
            self.assertIn(self.err_msg, str(err))
    else:
        self.assertIsNone(self.stack.validate())