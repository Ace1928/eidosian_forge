from __future__ import (absolute_import, division, print_function)
from os.path import basename
from ansible import constants as C
from ansible import context
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.color import colorize, hostcolor
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
def _preprocess_result(self, result):
    self.delegated_vars = result._result.get('_ansible_delegated_vars', None)
    self._handle_exception(result._result, use_stderr=self.get_option('display_failed_stderr'))
    self._handle_warnings(result._result)