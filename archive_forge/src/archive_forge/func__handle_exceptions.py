from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common._collections_compat import MutableMapping, MutableSequence
from ansible.plugins.callback.default import CallbackModule as CallbackModule_default
from ansible.utils.color import colorize, hostcolor
from ansible.utils.display import Display
import sys
def _handle_exceptions(self, result):
    if 'exception' in result:
        del result['exception']
        if self._display.verbosity == 1:
            return 'An exception occurred during task execution. To see the full traceback, use -vvv.'