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
def _list_plugins_from_paths(ptype, dirs, collection, depth=0):
    plugins = {}
    for path in dirs:
        display.debug("Searching '{0}'s '{1}' for {2} plugins".format(collection, path, ptype))
        b_path = to_bytes(path)
        if os.path.basename(b_path).startswith((b'.', b'__')):
            continue
        if os.path.exists(b_path):
            if os.path.isdir(b_path):
                bkey = ptype.lower()
                for plugin_file in os.listdir(b_path):
                    if plugin_file.startswith((b'.', b'__')):
                        continue
                    display.debug("Found possible plugin: '{0}'".format(plugin_file))
                    b_plugin, b_ext = os.path.splitext(plugin_file)
                    plugin = to_native(b_plugin)
                    full_path = os.path.join(b_path, plugin_file)
                    if os.path.isdir(full_path):
                        if collection in C.SYNTHETIC_COLLECTIONS:
                            if not os.path.exists(os.path.join(full_path, b'__init__.py')):
                                continue
                        plugins.update(_list_plugins_from_paths(ptype, [to_native(full_path)], collection, depth=depth + 1))
                    else:
                        if any([plugin in C.IGNORE_FILES, to_native(b_ext) in C.REJECT_EXTS, b_ext in (b'.yml', b'.yaml', b'.json'), plugin in IGNORE.get(bkey, ()), os.path.islink(full_path)]):
                            continue
                        if ptype in ('test', 'filter'):
                            try:
                                file_plugins = _list_j2_plugins_from_file(collection, full_path, ptype, plugin)
                            except KeyError as e:
                                display.warning('Skipping file %s: %s' % (full_path, to_native(e)))
                                continue
                            for plugin in file_plugins:
                                plugin_name = get_composite_name(collection, plugin.ansible_name, os.path.dirname(to_native(full_path)), depth)
                                plugins[plugin_name] = full_path
                        else:
                            plugin_name = get_composite_name(collection, plugin, os.path.dirname(to_native(full_path)), depth)
                            plugins[plugin_name] = full_path
            else:
                display.debug("Skip listing plugins in '{0}' as it is not a directory".format(path))
        else:
            display.debug("Skip listing plugins in '{0}' as it does not exist".format(path))
    return plugins