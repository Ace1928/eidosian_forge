from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
def _copy_attributes_to_actual(self):
    for attribute in self.readwrite_attrs:
        if attribute in self.attribute_values_processed:
            attribute_value = self.attribute_values_processed[attribute]
            if attribute_value is None:
                continue
            if attribute in self.json_encodes:
                attribute_value = json.JSONEncoder().encode(attribute_value).strip('"')
            setattr(self.actual, attribute, attribute_value)