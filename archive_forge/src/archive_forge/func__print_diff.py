from __future__ import (absolute_import, division, print_function)
import difflib
from ansible import constants as C
from ansible.plugins.callback import CallbackBase
from ansible.module_utils.common.text.converters import to_text
def _print_diff(self, diff, indent_level):
    if isinstance(diff, dict):
        try:
            diff = '\n'.join(difflib.unified_diff(diff['before'].splitlines(), diff['after'].splitlines(), fromfile=diff.get('before_header', 'new_file'), tofile=diff['after_header']))
        except AttributeError:
            diff = dict_diff(diff['before'], diff['after'])
    if diff:
        diff = colorize(str(diff), 'changed')
        print(self._indent_text(diff, indent_level + 4))