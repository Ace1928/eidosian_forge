import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _build_role_assignments_list(self, include_subtree=False):
    """List role assignments to user and groups on domains and projects.

        Return a list of all existing role assignments in the system, filtered
        by assignments attributes, if provided.

        If effective option is used and OS-INHERIT extension is enabled, the
        following functions will be applied:
        1) For any group role assignment on a target, replace it by a set of
        role assignments containing one for each user of that group on that
        target;
        2) For any inherited role assignment for an actor on a target, replace
        it by a set of role assignments for that actor on every project under
        that target.

        It means that, if effective mode is used, no group or domain inherited
        assignments will be present in the resultant list. Thus, combining
        effective with them is invalid.

        As a role assignment contains only one actor and one target, providing
        both user and group ids or domain and project ids is invalid as well.
        """
    params = flask.request.args
    include_names = self.query_filter_is_true('include_names')
    self._assert_domain_nand_project()
    self._assert_system_nand_domain()
    self._assert_system_nand_project()
    self._assert_user_nand_group()
    self._assert_effective_filters_if_needed()
    refs = PROVIDERS.assignment_api.list_role_assignments(role_id=params.get('role.id'), user_id=params.get('user.id'), group_id=params.get('group.id'), system=params.get('scope.system'), domain_id=params.get('scope.domain.id'), project_id=params.get('scope.project.id'), include_subtree=include_subtree, inherited=self._inherited, effective=self._effective, include_names=include_names)
    formatted_refs = [self._format_entity(ref) for ref in refs]
    return self.wrap_collection(formatted_refs)