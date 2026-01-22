import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
def _filter_ldap_result_by_attr(self, ldap_result, ldap_attr_name):
    attr = self.attribute_mapping[ldap_attr_name]
    if not attr:
        attr_name = '%s_%s_attribute' % (self.options_name, self.attribute_options_names[ldap_attr_name])
        raise ValueError('"%(attr)s" is not a valid value for "%(attr_name)s"' % {'attr': attr, 'attr_name': attr_name})
    result = []
    for obj in ldap_result:
        ldap_res_low_keys_dict = {k.lower(): v for k, v in obj[1].items()}
        result_attr_vals = ldap_res_low_keys_dict.get(attr.lower())
        if result_attr_vals:
            if result_attr_vals[0] and result_attr_vals[0].strip():
                result.append(obj)
    return result