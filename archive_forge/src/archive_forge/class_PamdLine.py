from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
class PamdLine(object):

    def __init__(self, line):
        self.line = line
        self.prev = None
        self.next = None

    @property
    def is_valid(self):
        if self.line.strip() == '':
            return True
        return False

    def validate(self):
        if not self.is_valid:
            return (False, 'Rule is not valid ' + self.line)
        return (True, 'Rule is valid ' + self.line)

    def matches(self, rule_type, rule_control, rule_path, rule_args=None):
        return False

    def __str__(self):
        return str(self.line)