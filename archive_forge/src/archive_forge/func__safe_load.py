from __future__ import (absolute_import, division, print_function)
import json
from yaml import YAMLError
from ansible.errors import AnsibleParserError
from ansible.errors.yaml_strings import YAML_SYNTAX_ERROR
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject
from ansible.parsing.ajson import AnsibleJSONDecoder
def _safe_load(stream, file_name=None, vault_secrets=None):
    """ Implements yaml.safe_load(), except using our custom loader class. """
    loader = AnsibleLoader(stream, file_name, vault_secrets)
    try:
        return loader.get_single_data()
    finally:
        try:
            loader.dispose()
        except AttributeError:
            pass