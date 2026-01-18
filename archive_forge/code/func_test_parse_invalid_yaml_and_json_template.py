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
def test_parse_invalid_yaml_and_json_template(self):
    tmpl_str = '{test'
    msg = 'line 1, column 1'
    self._parse_template(tmpl_str, msg)