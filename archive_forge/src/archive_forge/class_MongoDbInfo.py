from __future__ import absolute_import, division, print_function
from uuid import UUID
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
from ansible.module_utils.six import iteritems
from ansible_collections.community.mongodb.plugins.module_utils.mongodb_common import (
class MongoDbInfo:
    """Class for gathering MongoDB instance information.

    Args:
        module (AnsibleModule): Object of AnsibleModule class.
        client (pymongo): pymongo client object to interact with the database.
    """

    def __init__(self, module, client):
        self.module = module
        self.client = client
        self.admin_db = self.client.admin
        self.info = {'general': {}, 'databases': {}, 'total_size': {}, 'parameters': {}, 'users': {}, 'roles': {}}

    def get_info(self, filter_):
        """Get MongoDB instance information and return it based on filter_.

        Args:
            filter_ (list): List of collected subsets (e.g., general, users, etc.),
                when it is empty, return all available information.
        """
        self.__collect()
        inc_list = []
        exc_list = []
        if filter_:
            partial_info = {}
            for fi in filter_:
                if fi.lstrip('!') not in self.info:
                    self.module.warn("filter element '%s' is not allowable, ignored" % fi)
                    continue
                if fi[0] == '!':
                    exc_list.append(fi.lstrip('!'))
                else:
                    inc_list.append(fi)
            if inc_list:
                for i in self.info:
                    if i in inc_list:
                        partial_info[i] = self.info[i]
            else:
                for i in self.info:
                    if i not in exc_list:
                        partial_info[i] = self.info[i]
            return partial_info
        else:
            return self.info

    def __collect(self):
        """Collect information."""
        self.info['general'] = self.client.server_info()
        self.info['parameters'] = self.get_parameters_info()
        self.info['databases'], self.info['total_size'] = self.get_db_info()
        for dbname, val in iteritems(self.info['databases']):
            self.info['users'].update(self.get_users_info(dbname))
            self.info['roles'].update(self.get_roles_info(dbname))
        self.info = convert_bson_values_recur(self.info)

    def get_roles_info(self, dbname):
        """Gather information about roles.

        Args:
            dbname (str): Database name to get role info from.

        Returns a dictionary with role information for the given db.
        """
        db = self.client[dbname]
        result = db.command({'rolesInfo': 1, 'showBuiltinRoles': True})['roles']
        roles_dict = {}
        for elem in result:
            roles_dict[elem['role']] = {}
            for key, val in iteritems(elem):
                if key in ['role', 'db']:
                    continue
                roles_dict[elem['role']][key] = val
        return {dbname: roles_dict}

    def get_users_info(self, dbname):
        """Gather information about users.

        Args:
            dbname (str): Database name to get user info from.

        Returns a dictionary with user information for the given db.
        """
        db = self.client[dbname]
        result = db.command({'usersInfo': 1})['users']
        users_dict = {}
        for elem in result:
            users_dict[elem['user']] = {}
            for key, val in iteritems(elem):
                if key in ['user', 'db']:
                    continue
                if isinstance(val, UUID):
                    val = val.hex
                users_dict[elem['user']][key] = val
        return {dbname: users_dict}

    def get_db_info(self):
        """Gather information about databases.

        Returns a dictionary with database information.
        """
        result = self.admin_db.command({'listDatabases': 1})
        total_size = int(result['totalSize'])
        result = result['databases']
        db_dict = {}
        for elem in result:
            db_dict[elem['name']] = {}
            for key, val in iteritems(elem):
                if key == 'name':
                    continue
                if key == 'sizeOnDisk':
                    val = int(val)
                db_dict[elem['name']][key] = val
        return (db_dict, total_size)

    def get_parameters_info(self):
        """Gather parameters information.

        Returns a dictionary with parameters.
        """
        return self.admin_db.command({'getParameter': '*'})