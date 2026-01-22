from __future__ import absolute_import, division, print_function
import os
import traceback
from xml.etree.ElementTree import fromstring
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.six import iteritems, string_types
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import Template
def _extract_param(template, root, attrs, value):
    key = None
    when = attrs.get('when')
    conditional = '{%% if %s %%}True{%% else %%}False{%% endif %%}' % when
    param_to_xpath_map = attrs['items']
    if isinstance(value, Mapping):
        key = value.get('key', None)
        if key:
            value = value['values']
    entries = dict() if key else list()
    for element in root.findall(attrs['top']):
        entry = dict()
        item_dict = dict()
        for param, param_xpath in iteritems(param_to_xpath_map):
            fields = None
            try:
                fields = element.findall(param_xpath)
            except Exception:
                display.warning("Failed to evaluate value of '%s' with XPath '%s'.\nUnexpected error: %s." % (param, param_xpath, traceback.format_exc()))
            tags = param_xpath.split('/')
            if len(tags) and tags[-1].endswith(']'):
                if fields:
                    if len(fields) > 1:
                        item_dict[param] = [field.attrib for field in fields]
                    else:
                        item_dict[param] = fields[0].attrib
                else:
                    item_dict[param] = {}
            elif fields:
                if len(fields) > 1:
                    item_dict[param] = [field.text for field in fields]
                else:
                    item_dict[param] = fields[0].text
            else:
                item_dict[param] = None
        if isinstance(value, Mapping):
            for item_key, item_value in iteritems(value):
                entry[item_key] = template(item_value, {'item': item_dict})
        else:
            entry = template(value, {'item': item_dict})
        if key:
            expanded_key = template(key, {'item': item_dict})
            if when:
                if template(conditional, {'item': {'key': expanded_key, 'value': entry}}):
                    entries[expanded_key] = entry
            else:
                entries[expanded_key] = entry
        elif when:
            if template(conditional, {'item': entry}):
                entries.append(entry)
        else:
            entries.append(entry)
    return entries