from __future__ import (absolute_import, division, print_function)
import os
import ansible.constants as C
from ansible.errors import AnsibleParserError, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import string_types
from ansible.parsing.splitter import split_args
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.conditional import Conditional
from ansible.playbook.taggable import Taggable
from ansible.utils.collection_loader import AnsibleCollectionConfig
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path, _get_collection_playbook_path
from ansible.template import Templar
from ansible.utils.display import Display

        Splits the playbook import line up into filename and parameters
        