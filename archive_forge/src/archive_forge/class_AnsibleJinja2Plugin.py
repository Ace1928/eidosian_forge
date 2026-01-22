from __future__ import (absolute_import, division, print_function)
from abc import ABC
import types
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
class AnsibleJinja2Plugin(AnsiblePlugin):

    def __init__(self, function):
        super(AnsibleJinja2Plugin, self).__init__()
        self._function = function

    @property
    def plugin_type(self):
        return self.__class__.__name__.lower().replace('ansiblejinja2', '')

    def _no_options(self, *args, **kwargs):
        raise NotImplementedError()
    has_option = get_option = get_options = option_definitions = set_option = set_options = _no_options

    @property
    def j2_function(self):
        return self._function