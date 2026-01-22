from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import to_list
from ansible_collections.ansible.utils.plugins.plugin_utils.base.validate import ValidateBase
def _check_drafts(self):
    """For every possible draft check if our jsonschema version supports it and exchange the class names with
        the actual classes. If it is not supported the draft is removed from the list.
        """
    for draft in list(self._JSONSCHEMA_DRAFTS.keys()):
        draft_config = self._JSONSCHEMA_DRAFTS[draft]
        try:
            validator_class = getattr(jsonschema, draft_config['validator_name'])
        except AttributeError:
            display.vvv('jsonschema draft "{draft}" not supported in this version'.format(draft=draft))
            del self._JSONSCHEMA_DRAFTS[draft]
            continue
        draft_config['validator'] = validator_class
        try:
            format_checker_class = validator_class.FORMAT_CHECKER
        except AttributeError:
            format_checker_class = getattr(jsonschema, draft_config['format_checker_name'])
        draft_config['format_checker'] = format_checker_class