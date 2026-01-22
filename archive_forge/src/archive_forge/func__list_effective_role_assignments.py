import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _list_effective_role_assignments(self, role_id, user_id, group_id, domain_id, project_id, subtree_ids, inherited, source_from_group_ids, strip_domain_roles):
    """List role assignments in effective mode.

        When using effective mode, besides the direct assignments, the indirect
        ones that come from grouping or inheritance are retrieved and will then
        be expanded.

        The resulting list of assignments will be filtered by the provided
        parameters. If subtree_ids is not None, then we also want to include
        all subtree_ids in the filter as well. Since we are in effective mode,
        group can never act as a filter (since group assignments are expanded
        into user roles) and domain can only be filter if we want non-inherited
        assignments, since domains can't inherit assignments.

        The goal of this method is to only ask the driver for those
        assignments as could effect the result based on the parameter filters
        specified, hence avoiding retrieving a huge list.

        """

    def list_role_assignments_for_actor(role_id, inherited, user_id=None, group_ids=None, project_id=None, subtree_ids=None, domain_id=None):
        """List role assignments for actor on target.

            List direct and indirect assignments for an actor, optionally
            for a given target (i.e. projects or domain).

            :param role_id: List for a specific role, can be None meaning all
                            roles
            :param inherited: Indicates whether inherited assignments or only
                              direct assignments are required.  If None, then
                              both are required.
            :param user_id: If not None, list only assignments that affect this
                            user.
            :param group_ids: A list of groups required. Only one of user_id
                              and group_ids can be specified
            :param project_id: If specified, only include those assignments
                               that affect at least this project, with
                               additionally any projects specified in
                               subtree_ids
            :param subtree_ids: The list of projects in the subtree. If
                                specified, also include those assignments that
                                affect these projects. These projects are
                                guaranteed to be in the same domain as the
                                project specified in project_id. subtree_ids
                                can only be specified if project_id has also
                                been specified.
            :param domain_id: If specified, only include those assignments
                              that affect this domain - by definition this will
                              not include any inherited assignments

            :returns: List of assignments matching the criteria. Any inherited
                      or group assignments that could affect the resulting
                      response are included.

            """
        project_ids_of_interest = None
        if project_id:
            if subtree_ids:
                project_ids_of_interest = subtree_ids + [project_id]
            else:
                project_ids_of_interest = [project_id]
        non_inherited_refs = []
        if inherited is False or inherited is None:
            non_inherited_refs = self.driver.list_role_assignments(role_id=role_id, domain_id=domain_id, project_ids=project_ids_of_interest, user_id=user_id, group_ids=group_ids, inherited_to_projects=False)
        inherited_refs = []
        if inherited is True or inherited is None:
            if project_id:
                proj_domain_id = PROVIDERS.resource_api.get_project(project_id)['domain_id']
                inherited_refs += self.driver.list_role_assignments(role_id=role_id, domain_id=proj_domain_id, user_id=user_id, group_ids=group_ids, inherited_to_projects=True)
                source_ids = [project['id'] for project in PROVIDERS.resource_api.list_project_parents(project_id)]
                if subtree_ids:
                    source_ids += project_ids_of_interest
                if source_ids:
                    inherited_refs += self.driver.list_role_assignments(role_id=role_id, project_ids=source_ids, user_id=user_id, group_ids=group_ids, inherited_to_projects=True)
            else:
                inherited_refs = self.driver.list_role_assignments(role_id=role_id, user_id=user_id, group_ids=group_ids, inherited_to_projects=True)
        return non_inherited_refs + inherited_refs
    if group_id or (domain_id and inherited):
        return []
    if user_id and source_from_group_ids:
        msg = _('Cannot list assignments sourced from groups and filtered by user ID.')
        raise exception.UnexpectedError(msg)
    inherited = False if domain_id else inherited
    direct_refs = list_role_assignments_for_actor(role_id=None, user_id=user_id, group_ids=source_from_group_ids, project_id=project_id, subtree_ids=subtree_ids, domain_id=domain_id, inherited=inherited)
    group_refs = []
    if not source_from_group_ids and user_id:
        group_ids = self._get_group_ids_for_user_id(user_id)
        if group_ids:
            group_refs = list_role_assignments_for_actor(role_id=None, project_id=project_id, subtree_ids=subtree_ids, group_ids=group_ids, domain_id=domain_id, inherited=inherited)
    refs = []
    expand_groups = source_from_group_ids is None
    for ref in direct_refs + group_refs:
        refs += self._expand_indirect_assignment(ref, user_id, project_id, subtree_ids, expand_groups)
    refs = self.add_implied_roles(refs)
    if strip_domain_roles:
        refs = self._strip_domain_roles(refs)
    if role_id:
        refs = self._filter_by_role_id(role_id, refs)
    return refs