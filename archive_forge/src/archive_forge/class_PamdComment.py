from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
import os
import re
from tempfile import NamedTemporaryFile
from datetime import datetime
class PamdComment(PamdLine):

    def __init__(self, line):
        super(PamdComment, self).__init__(line)

    @property
    def is_valid(self):
        if self.line.startswith('#'):
            return True
        return False