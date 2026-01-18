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
def test_parse_string_template(self):
    tmpl_str = 'just string'
    msg = 'The template is not a JSON object or YAML mapping.'
    self._parse_template(tmpl_str, msg)