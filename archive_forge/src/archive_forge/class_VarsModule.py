from __future__ import (absolute_import, division, print_function)
import os
from ansible.errors import AnsibleParserError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.vars import BaseVarsPlugin
from ansible.utils.path import basedir
from ansible.inventory.group import InventoryObjectType
from ansible.utils.vars import combine_vars
class VarsModule(BaseVarsPlugin):
    REQUIRES_ENABLED = True
    is_stateless = True

    def load_found_files(self, loader, data, found_files):
        for found in found_files:
            new_data = loader.load_from_file(found, cache=True, unsafe=True)
            if new_data:
                data = combine_vars(data, new_data)
        return data

    def get_vars(self, loader, path, entities, cache=True):
        """ parses the inventory file """
        if not isinstance(entities, list):
            entities = [entities]
        try:
            realpath_basedir = CANONICAL_PATHS[path]
        except KeyError:
            CANONICAL_PATHS[path] = realpath_basedir = os.path.realpath(basedir(path))
        data = {}
        for entity in entities:
            try:
                entity_name = entity.name
            except AttributeError:
                raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
            try:
                first_char = entity_name[0]
            except (TypeError, IndexError, KeyError):
                raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
            if first_char != os.path.sep:
                try:
                    found_files = []
                    try:
                        entity_type = entity.base_type
                    except AttributeError:
                        raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
                    if entity_type is InventoryObjectType.HOST:
                        subdir = 'host_vars'
                    elif entity_type is InventoryObjectType.GROUP:
                        subdir = 'group_vars'
                    else:
                        raise AnsibleParserError('Supplied entity must be Host or Group, got %s instead' % type(entity))
                    if cache:
                        try:
                            opath = PATH_CACHE[realpath_basedir, subdir]
                        except KeyError:
                            opath = PATH_CACHE[realpath_basedir, subdir] = os.path.join(realpath_basedir, subdir)
                        if opath in NAK:
                            continue
                        key = '%s.%s' % (entity_name, opath)
                        if key in FOUND:
                            data = self.load_found_files(loader, data, FOUND[key])
                            continue
                    else:
                        opath = PATH_CACHE[realpath_basedir, subdir] = os.path.join(realpath_basedir, subdir)
                    if os.path.isdir(opath):
                        self._display.debug('\tprocessing dir %s' % opath)
                        FOUND[key] = found_files = loader.find_vars_files(opath, entity_name)
                    elif not os.path.exists(opath):
                        NAK.add(opath)
                    else:
                        self._display.warning('Found %s that is not a directory, skipping: %s' % (subdir, opath))
                        NAK.add(opath)
                    data = self.load_found_files(loader, data, found_files)
                except Exception as e:
                    raise AnsibleParserError(to_native(e))
        return data