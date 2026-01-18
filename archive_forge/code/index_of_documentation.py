from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils._text import to_native
from ansible.module_utils.six import integer_types, string_types
from jinja2.exceptions import TemplateSyntaxError
Find the index or indices of entries in list of objects"

    :param data: The data passed in (data|index_of(...))
    :type data: unknown
    :param test: the test to use
    :type test: jinja2 test
    :param value: The value to use for the test
    :type value: unknown
    :param key: The key to use when a list of dicts is passed
    :type key: valid key type
    :param want_list: always return a list, even if 1 index
    :type want_list: bool
    :param fail_on_missing: Should we fail if key not found?
    :type fail_on_missing: bool
    :param tests: The jinja tests from the current environment
    :type tests: ansible.template.JinjaPluginIntercept
    