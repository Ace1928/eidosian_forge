from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _get_assignment_types(self, user, group, project, domain):
    """Return a list of role assignment types based on provided entities.

        If one of user or group (the "actor") as well as one of project or
        domain (the "target") are provided, the list will contain the role
        assignment type for that specific pair of actor and target.

        If only an actor or target is provided, the list will contain the
        role assignment types that satisfy the specified entity.

        For example, if user and project are provided, the return will be:

            [AssignmentType.USER_PROJECT]

        However, if only user was provided, the return would be:

            [AssignmentType.USER_PROJECT, AssignmentType.USER_DOMAIN]

        It is not expected that user and group (or project and domain) are
        specified - but if they are, the most fine-grained value will be
        chosen (i.e. user over group, project over domain).

        """
    actor_types = []
    if user:
        actor_types = self._get_user_assignment_types()
    elif group:
        actor_types = self._get_group_assignment_types()
    target_types = []
    if project:
        target_types = self._get_project_assignment_types()
    elif domain:
        target_types = self._get_domain_assignment_types()
    if actor_types and target_types:
        return list(set(actor_types).intersection(target_types))
    return actor_types or target_types