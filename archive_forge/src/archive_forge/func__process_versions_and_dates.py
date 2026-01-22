from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping, MutableSet, MutableSequence
from pathlib import Path
from ansible import constants as C
from ansible.release import __version__ as ansible_version
from ansible.errors import AnsibleError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.plugin_docs import read_docstring
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def _process_versions_and_dates(fragment, is_module, return_docs, callback):

    def process_deprecation(deprecation, top_level=False):
        collection_name = 'removed_from_collection' if top_level else 'collection_name'
        if not isinstance(deprecation, MutableMapping):
            return
        if (is_module or top_level) and 'removed_in' in deprecation:
            callback(deprecation, 'removed_in', collection_name)
        if 'removed_at_date' in deprecation:
            callback(deprecation, 'removed_at_date', collection_name)
        if not (is_module or top_level) and 'version' in deprecation:
            callback(deprecation, 'version', collection_name)

    def process_option_specifiers(specifiers):
        for specifier in specifiers:
            if not isinstance(specifier, MutableMapping):
                continue
            if 'version_added' in specifier:
                callback(specifier, 'version_added', 'version_added_collection')
            if isinstance(specifier.get('deprecated'), MutableMapping):
                process_deprecation(specifier['deprecated'])

    def process_options(options):
        for option in options.values():
            if not isinstance(option, MutableMapping):
                continue
            if 'version_added' in option:
                callback(option, 'version_added', 'version_added_collection')
            if not is_module:
                if isinstance(option.get('env'), list):
                    process_option_specifiers(option['env'])
                if isinstance(option.get('ini'), list):
                    process_option_specifiers(option['ini'])
                if isinstance(option.get('vars'), list):
                    process_option_specifiers(option['vars'])
                if isinstance(option.get('deprecated'), MutableMapping):
                    process_deprecation(option['deprecated'])
            if isinstance(option.get('suboptions'), MutableMapping):
                process_options(option['suboptions'])

    def process_return_values(return_values):
        for return_value in return_values.values():
            if not isinstance(return_value, MutableMapping):
                continue
            if 'version_added' in return_value:
                callback(return_value, 'version_added', 'version_added_collection')
            if isinstance(return_value.get('contains'), MutableMapping):
                process_return_values(return_value['contains'])

    def process_attributes(attributes):
        for attribute in attributes.values():
            if not isinstance(attribute, MutableMapping):
                continue
            if 'version_added' in attribute:
                callback(attribute, 'version_added', 'version_added_collection')
    if not fragment:
        return
    if return_docs:
        process_return_values(fragment)
        return
    if 'version_added' in fragment:
        callback(fragment, 'version_added', 'version_added_collection')
    if isinstance(fragment.get('deprecated'), MutableMapping):
        process_deprecation(fragment['deprecated'], top_level=True)
    if isinstance(fragment.get('options'), MutableMapping):
        process_options(fragment['options'])
    if isinstance(fragment.get('attributes'), MutableMapping):
        process_attributes(fragment['attributes'])