from __future__ import (absolute_import, division, print_function)
import sys
from contextlib import contextmanager
from ansible.template import Templar
from ansible.vars.manager import VariableManager
from ansible.plugins.callback.default import CallbackModule as Default
from ansible.module_utils.common.text.converters import to_text
def _parent_has_callback(self):
    return hasattr(super(CallbackModule, self), sys._getframe(1).f_code.co_name)