import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
Format an assignment entity for API response.

        The driver layer returns entities as dicts containing the ids of the
        actor (e.g. user or group), target (e.g. domain or project) and role.
        If it is an inherited role, then this is also indicated. Examples:

        For a non-inherited expanded assignment from group membership:
        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'group_id': group_id}}

        or, for a project inherited role:

        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'project_id': parent_id}}

        or, for a role that was implied by a prior role:

        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'role_id': prior role_id}}

        It is possible to deduce if a role assignment came from group
        membership if it has both 'user_id' in the main body of the dict and
        'group_id' in the 'indirect' subdict, as well as it is possible to
        deduce if it has come from inheritance if it contains both a
        'project_id' in the main body of the dict and 'parent_id' in the
        'indirect' subdict.

        This function maps this into the format to be returned via the API,
        e.g. for the second example above:

        {
            'user': {
                {'id': user_id}
            },
            'scope': {
                'project': {
                    {'id': project_id}
                },
                'OS-INHERIT:inherited_to': 'projects'
            },
            'role': {
                {'id': role_id}
            },
            'links': {
                'assignment': '/OS-INHERIT/projects/parent_id/users/user_id/'
                              'roles/role_id/inherited_to_projects'
            }
        }

        