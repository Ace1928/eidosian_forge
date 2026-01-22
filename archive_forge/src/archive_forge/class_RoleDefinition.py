from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import string_types
from ansible.parsing.yaml.objects import AnsibleBaseYAMLObject, AnsibleMapping
from ansible.playbook.attribute import NonInheritableFieldAttribute
from ansible.playbook.base import Base
from ansible.playbook.collectionsearch import CollectionSearch
from ansible.playbook.conditional import Conditional
from ansible.playbook.taggable import Taggable
from ansible.template import Templar
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_role_path
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
class RoleDefinition(Base, Conditional, Taggable, CollectionSearch):
    role = NonInheritableFieldAttribute(isa='string')

    def __init__(self, play=None, role_basedir=None, variable_manager=None, loader=None, collection_list=None):
        super(RoleDefinition, self).__init__()
        self._play = play
        self._variable_manager = variable_manager
        self._loader = loader
        self._role_path = None
        self._role_collection = None
        self._role_basedir = role_basedir
        self._role_params = dict()
        self._collection_list = collection_list

    @staticmethod
    def load(data, variable_manager=None, loader=None):
        raise AnsibleError('not implemented')

    def preprocess_data(self, ds):
        if isinstance(ds, int):
            ds = '%s' % ds
        if not isinstance(ds, dict) and (not isinstance(ds, string_types)) and (not isinstance(ds, AnsibleBaseYAMLObject)):
            raise AnsibleAssertionError()
        if isinstance(ds, dict):
            ds = super(RoleDefinition, self).preprocess_data(ds)
        self._ds = ds
        new_ds = AnsibleMapping()
        if isinstance(ds, AnsibleBaseYAMLObject):
            new_ds.ansible_pos = ds.ansible_pos
        role_name = self._load_role_name(ds)
        role_name, role_path = self._load_role_path(role_name)
        if isinstance(ds, dict):
            new_role_def, role_params = self._split_role_params(ds)
            new_ds |= new_role_def
            self._role_params = role_params
        new_ds['role'] = role_name
        self._role_path = role_path
        return new_ds

    def _load_role_name(self, ds):
        """
        Returns the role name (either the role: or name: field) from
        the role definition, or (when the role definition is a simple
        string), just that string
        """
        if isinstance(ds, string_types):
            return ds
        role_name = ds.get('role', ds.get('name'))
        if not role_name or not isinstance(role_name, string_types):
            raise AnsibleError('role definitions must contain a role name', obj=ds)
        if self._variable_manager:
            all_vars = self._variable_manager.get_vars(play=self._play)
            templar = Templar(loader=self._loader, variables=all_vars)
            role_name = templar.template(role_name)
        return role_name

    def _load_role_path(self, role_name):
        """
        the 'role', as specified in the ds (or as a bare string), can either
        be a simple name or a full path. If it is a full path, we use the
        basename as the role name, otherwise we take the name as-given and
        append it to the default role path
        """
        if self._variable_manager is not None:
            all_vars = self._variable_manager.get_vars(play=self._play)
        else:
            all_vars = dict()
        templar = Templar(loader=self._loader, variables=all_vars)
        role_name = templar.template(role_name)
        role_tuple = None
        if self._collection_list or AnsibleCollectionRef.is_valid_fqcr(role_name):
            role_tuple = _get_collection_role_path(role_name, self._collection_list)
        if role_tuple:
            self._role_collection = role_tuple[2]
            return role_tuple[0:2]
        role_search_paths = [os.path.join(self._loader.get_basedir(), u'roles')]
        if C.DEFAULT_ROLES_PATH:
            role_search_paths.extend(C.DEFAULT_ROLES_PATH)
        if self._role_basedir:
            role_search_paths.append(self._role_basedir)
        role_search_paths.append(self._loader.get_basedir())
        for path in role_search_paths:
            path = templar.template(path)
            role_path = unfrackpath(os.path.join(path, role_name))
            if self._loader.path_exists(role_path):
                return (role_name, role_path)
        role_path = unfrackpath(role_name)
        if self._loader.path_exists(role_path):
            role_name = os.path.basename(role_name)
            return (role_name, role_path)
        searches = (self._collection_list or []) + role_search_paths
        raise AnsibleError("the role '%s' was not found in %s" % (role_name, ':'.join(searches)), obj=self._ds)

    def _split_role_params(self, ds):
        """
        Splits any random role params off from the role spec and store
        them in a dictionary of params for parsing later
        """
        role_def = dict()
        role_params = dict()
        base_attribute_names = frozenset(self.fattributes)
        for key, value in ds.items():
            if key not in base_attribute_names:
                role_params[key] = value
            else:
                role_def[key] = value
        return (role_def, role_params)

    def get_role_params(self):
        return self._role_params.copy()

    def get_role_path(self):
        return self._role_path

    def get_name(self, include_role_fqcn=True):
        if include_role_fqcn:
            return '.'.join((x for x in (self._role_collection, self.role) if x))
        return self.role