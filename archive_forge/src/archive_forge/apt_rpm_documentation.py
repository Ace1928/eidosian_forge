from __future__ import absolute_import, division, print_function
import os
import re
import traceback
from ansible.module_utils.basic import (
from ansible.module_utils.common.text.converters import to_native
return package name of a local rpm passed in.
    Inspired by ansible.builtin.yum