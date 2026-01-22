from __future__ import absolute_import, division, print_function
from datetime import timedelta
from decimal import Decimal
from os import environ
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import iteritems
from ansible_collections.community.postgresql.plugins.module_utils.version import \
class PgMembership(object):

    def __init__(self, module, cursor, groups, target_roles, fail_on_role=True):
        self.module = module
        self.cursor = cursor
        self.target_roles = [r.strip() for r in target_roles]
        self.groups = [r.strip() for r in groups]
        self.executed_queries = []
        self.granted = {}
        self.revoked = {}
        self.fail_on_role = fail_on_role
        self.non_existent_roles = []
        self.changed = False
        self.__check_roles_exist()

    def grant(self):
        for group in self.groups:
            self.granted[group] = []
            for role in self.target_roles:
                role_obj = PgRole(self.module, self.cursor, role)
                if group in role_obj.memberof:
                    continue
                query = 'GRANT "%s" TO "%s"' % (group, role)
                self.changed = exec_sql(self, query, return_bool=True)
                if self.changed:
                    self.granted[group].append(role)
        return self.changed

    def revoke(self):
        for group in self.groups:
            self.revoked[group] = []
            for role in self.target_roles:
                role_obj = PgRole(self.module, self.cursor, role)
                if group not in role_obj.memberof:
                    continue
                query = 'REVOKE "%s" FROM "%s"' % (group, role)
                self.changed = exec_sql(self, query, return_bool=True)
                if self.changed:
                    self.revoked[group].append(role)
        return self.changed

    def match(self):
        for role in self.target_roles:
            role_obj = PgRole(self.module, self.cursor, role)
            desired_groups = set(self.groups)
            current_groups = set(role_obj.memberof)
            groups_to_revoke = current_groups - desired_groups
            for group in groups_to_revoke:
                query = 'REVOKE "%s" FROM "%s"' % (group, role)
                self.changed = exec_sql(self, query, return_bool=True)
                if group in self.revoked:
                    self.revoked[group].append(role)
                else:
                    self.revoked[group] = [role]
            groups_to_grant = desired_groups - current_groups
            for group in groups_to_grant:
                query = 'GRANT "%s" TO "%s"' % (group, role)
                self.changed = exec_sql(self, query, return_bool=True)
                if group in self.granted:
                    self.granted[group].append(role)
                else:
                    self.granted[group] = [role]
        return self.changed

    def __check_roles_exist(self):
        if self.groups:
            existent_groups = self.__roles_exist(self.groups)
            for group in self.groups:
                if group not in existent_groups:
                    if self.fail_on_role:
                        self.module.fail_json(msg='Role %s does not exist' % group)
                    else:
                        self.module.warn('Role %s does not exist, pass' % group)
                        self.non_existent_roles.append(group)
        existent_roles = self.__roles_exist(self.target_roles)
        for role in self.target_roles:
            if role not in existent_roles:
                if self.fail_on_role:
                    self.module.fail_json(msg='Role %s does not exist' % role)
                else:
                    self.module.warn('Role %s does not exist, pass' % role)
                if role not in self.groups:
                    self.non_existent_roles.append(role)
                elif self.fail_on_role:
                    self.module.exit_json(msg="Role role '%s' is a member of role '%s'" % (role, role))
                else:
                    self.module.warn("Role role '%s' is a member of role '%s', pass" % (role, role))
        if self.groups:
            self.groups = [g for g in self.groups if g not in self.non_existent_roles]
        self.target_roles = [r for r in self.target_roles if r not in self.non_existent_roles]

    def __roles_exist(self, roles):
        tmp = ["'" + x + "'" for x in roles]
        query = 'SELECT rolname FROM pg_roles WHERE rolname IN (%s)' % ','.join(tmp)
        return [x['rolname'] for x in exec_sql(self, query, add_to_executed=False)]