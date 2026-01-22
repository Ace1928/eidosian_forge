from __future__ import absolute_import, division, print_function
from datetime import datetime
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import binary_type, text_type
class OSXDefaultsException(Exception):

    def __init__(self, msg):
        self.message = msg