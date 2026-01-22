from __future__ import absolute_import, division, print_function
import json
import re
import sys
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils._text import to_native
class ConfigProxy(object):

    def __init__(self, actual, client, attribute_values_dict, readwrite_attrs, transforms=None, readonly_attrs=None, immutable_attrs=None, json_encodes=None):
        transforms = {} if transforms is None else transforms
        readonly_attrs = [] if readonly_attrs is None else readonly_attrs
        immutable_attrs = [] if immutable_attrs is None else immutable_attrs
        json_encodes = [] if json_encodes is None else json_encodes
        self.actual = actual
        self.client = client
        self.attribute_values_dict = attribute_values_dict
        self.readwrite_attrs = readwrite_attrs
        self.readonly_attrs = readonly_attrs
        self.immutable_attrs = immutable_attrs
        self.json_encodes = json_encodes
        self.transforms = transforms
        self.attribute_values_processed = {}
        for attribute, value in self.attribute_values_dict.items():
            if value is None:
                continue
            if attribute in transforms:
                for transform in self.transforms[attribute]:
                    if transform == 'bool_yes_no':
                        if value is True:
                            value = 'YES'
                        elif value is False:
                            value = 'NO'
                    elif transform == 'bool_on_off':
                        if value is True:
                            value = 'ON'
                        elif value is False:
                            value = 'OFF'
                    elif callable(transform):
                        value = transform(value)
                    else:
                        raise Exception('Invalid transform %s' % transform)
            self.attribute_values_processed[attribute] = value
        self._copy_attributes_to_actual()

    def _copy_attributes_to_actual(self):
        for attribute in self.readwrite_attrs:
            if attribute in self.attribute_values_processed:
                attribute_value = self.attribute_values_processed[attribute]
                if attribute_value is None:
                    continue
                if attribute in self.json_encodes:
                    attribute_value = json.JSONEncoder().encode(attribute_value).strip('"')
                setattr(self.actual, attribute, attribute_value)

    def __getattr__(self, name):
        if name in self.attribute_values_dict:
            return self.attribute_values_dict[name]
        else:
            raise AttributeError('No attribute %s found' % name)

    def add(self):
        self.actual.__class__.add(self.client, self.actual)

    def update(self):
        return self.actual.__class__.update(self.client, self.actual)

    def delete(self):
        self.actual.__class__.delete(self.client, self.actual)

    def get(self, *args, **kwargs):
        result = self.actual.__class__.get(self.client, *args, **kwargs)
        return result

    def has_equal_attributes(self, other):
        if self.diff_object(other) == {}:
            return True
        else:
            return False

    def diff_object(self, other):
        diff_dict = {}
        for attribute in self.attribute_values_processed:
            if attribute not in self.readwrite_attrs:
                continue
            if self.attribute_values_processed[attribute] is None:
                continue
            if hasattr(other, attribute):
                attribute_value = getattr(other, attribute)
            else:
                diff_dict[attribute] = 'missing from other'
                continue
            param_type = self.attribute_values_processed[attribute].__class__
            if attribute_value is None or param_type(attribute_value) != self.attribute_values_processed[attribute]:
                str_tuple = (type(self.attribute_values_processed[attribute]), self.attribute_values_processed[attribute], type(attribute_value), attribute_value)
                diff_dict[attribute] = 'difference. ours: (%s) %s other: (%s) %s' % str_tuple
        return diff_dict

    def get_actual_rw_attributes(self, filter='name'):
        if self.actual.__class__.count_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter])) == 0:
            return {}
        server_list = self.actual.__class__.get_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter]))
        actual_instance = server_list[0]
        ret_val = {}
        for attribute in self.readwrite_attrs:
            if not hasattr(actual_instance, attribute):
                continue
            ret_val[attribute] = getattr(actual_instance, attribute)
        return ret_val

    def get_actual_ro_attributes(self, filter='name'):
        if self.actual.__class__.count_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter])) == 0:
            return {}
        server_list = self.actual.__class__.get_filtered(self.client, '%s:%s' % (filter, self.attribute_values_dict[filter]))
        actual_instance = server_list[0]
        ret_val = {}
        for attribute in self.readonly_attrs:
            if not hasattr(actual_instance, attribute):
                continue
            ret_val[attribute] = getattr(actual_instance, attribute)
        return ret_val

    def get_missing_rw_attributes(self):
        return list(set(self.readwrite_attrs) - set(self.get_actual_rw_attributes().keys()))

    def get_missing_ro_attributes(self):
        return list(set(self.readonly_attrs) - set(self.get_actual_ro_attributes().keys()))