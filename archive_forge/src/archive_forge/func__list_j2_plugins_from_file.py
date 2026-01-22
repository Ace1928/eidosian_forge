from __future__ import (absolute_import, division, print_function)
import os
from ansible import context
from ansible import constants as C
from ansible.collections.list import list_collections
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible.plugins import loader
from ansible.utils.display import Display
from ansible.utils.collection_loader._collection_finder import _get_collection_path
def _list_j2_plugins_from_file(collection, plugin_path, ptype, plugin_name):
    ploader = getattr(loader, '{0}_loader'.format(ptype))
    file_plugins = ploader.get_contained_plugins(collection, plugin_path, plugin_name)
    return file_plugins