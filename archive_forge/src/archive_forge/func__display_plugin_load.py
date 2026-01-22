from __future__ import (absolute_import, division, print_function)
import glob
import os
import os.path
import pkgutil
import sys
import warnings
from collections import defaultdict, namedtuple
from traceback import format_exc
import ansible.module_utils.compat.typing as t
from .filter import AnsibleJinja2Filter
from .test import AnsibleJinja2Test
from ansible import __version__ as ansible_version
from ansible import constants as C
from ansible.errors import AnsibleError, AnsiblePluginCircularRedirect, AnsiblePluginRemovedError, AnsibleCollectionUnsupportedVersionError
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.module_utils.compat.importlib import import_module
from ansible.module_utils.six import string_types
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.plugins import get_plugin_class, MODULE_CACHE, PATH_CACHE, PLUGIN_PATH_CACHE
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder, _get_collection_metadata
from ansible.utils.display import Display
from ansible.utils.plugin_docs import add_fragments
from ansible.utils.unsafe_proxy import _is_unsafe
import importlib.util
def _display_plugin_load(self, class_name, name, searched_paths, path, found_in_cache=None, class_only=None):
    """ formats data to display debug info for plugin loading, also avoids processing unless really needed """
    if C.DEFAULT_DEBUG:
        msg = "Loading %s '%s' from %s" % (class_name, os.path.basename(name), path)
        if len(searched_paths) > 1:
            msg = '%s (searched paths: %s)' % (msg, self.format_paths(searched_paths))
        if found_in_cache or class_only:
            msg = '%s (found_in_cache=%s, class_only=%s)' % (msg, found_in_cache, class_only)
        display.debug(msg)