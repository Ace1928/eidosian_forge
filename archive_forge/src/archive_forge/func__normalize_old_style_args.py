from __future__ import (absolute_import, division, print_function)
import ansible.constants as C
from ansible.errors import AnsibleParserError, AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_text
from ansible.parsing.splitter import parse_kv, split_args
from ansible.plugins.loader import module_loader, action_loader
from ansible.template import Templar
from ansible.utils.fqcn import add_internal_fqcns
from ansible.utils.sentinel import Sentinel
def _normalize_old_style_args(self, thing):
    """
        deals with fuzziness in old-style (action/local_action) module invocations
        returns tuple of (module_name, dictionary_args)

        possible example inputs:
           { 'shell' : 'echo hi' }
           'shell echo hi'
           {'module': 'ec2', 'x': 1 }
        standardized outputs like:
           ('ec2', { 'x': 1} )
        """
    action = None
    args = None
    if isinstance(thing, dict):
        thing = thing.copy()
        if 'module' in thing:
            action, module_args = self._split_module_string(thing['module'])
            args = thing.copy()
            check_raw = action in FREEFORM_ACTIONS
            args.update(parse_kv(module_args, check_raw=check_raw))
            del args['module']
    elif isinstance(thing, string_types):
        action, args = self._split_module_string(thing)
        check_raw = action in FREEFORM_ACTIONS
        args = parse_kv(args, check_raw=check_raw)
    else:
        raise AnsibleParserError('unexpected parameter type in action: %s' % type(thing), obj=self._task_ds)
    return (action, args)