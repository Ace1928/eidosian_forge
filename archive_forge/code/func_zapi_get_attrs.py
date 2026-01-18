from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import re
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
def zapi_get_attrs(self, na_element, attr_dict, result):
    """ Retrieve a list of attributes from na_elements
        see na_ontap_volume for an example.
        na_element: xml element as returned by ZAPI.
        attr_dict:
            A dict of dict, with format:
                key: dict(key_list, required=False, default=None, convert_to=None, omitnone=False)
            The keys are used to index a result dictionary, values are read from a ZAPI object indexed by key_list.
            If required is True, an error is reported if a key in key_list is not found.
            If required is False and the value is not found, uses default as the value.
            If convert_to is set to str, bool, int, the ZAPI value is converted from str to the desired type.
            I'm not sure there is much value in omitnone, but it preserves backward compatibility.
            When the value is None, if omitnone is False, a None value is recorded, if True, the key is not set.
        result: an existing dictionary.  keys are added or updated based on attrs.

        Errors: fail_json is called for:
            - a key is not found and required=True,
            - a format conversion error
        """
    for key, kwargs in attr_dict.items():
        omitnone = kwargs.pop('omitnone', False)
        value = self.zapi_get_value(na_element, **kwargs)
        if value is not None or not omitnone:
            result[key] = value