from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def zapi_get_value(self, na_element, key_list, required=False, default=None, convert_to=None):
    """ read a value from na_element using key_list

            If required is True, an error is reported if a key in key_list is not found.
            If required is False and the value is not found, uses default as the value.
            If convert_to is set to str, bool, int, the ZAPI value is converted from str to the desired type.
                suported values: None, the python types int, str, bool, special 'bool_online'

        Errors: fail_json is called for:
            - a key is not found and required=True,
            - a format conversion error
        """
    saved_key_list = list(key_list)
    try:
        value = self.safe_get(na_element, key_list, allow_sparse_dict=not required)
    except (KeyError, TypeError) as exc:
        error = exc
    else:
        value, error = self.convert_value(value, convert_to) if value is not None else (default, None)
    if error:
        self.ansible_module.fail_json(msg='Error reading %s from %s: %s' % (saved_key_list, na_element.to_string(), error))
    return value