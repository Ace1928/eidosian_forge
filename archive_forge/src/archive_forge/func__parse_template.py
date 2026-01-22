import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
def _parse_template(self, tmpl_str, msg_str):
    parse_ex = self.assertRaises(ValueError, template_format.parse, tmpl_str)
    self.assertIn(msg_str, str(parse_ex))