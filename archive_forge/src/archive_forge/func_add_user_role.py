from keystoneclient import base
def add_user_role(self, user, role, tenant=None):
    """Add a role to a user.

        If tenant is specified, the role is added just for that tenant,
        otherwise the role is added globally.
        """
    user_id = base.getid(user)
    role_id = base.getid(role)
    if tenant:
        route = '/tenants/%s/users/%s/roles/OS-KSADM/%s'
        params = (base.getid(tenant), user_id, role_id)
        return self._update(route % params, None, 'role')
    else:
        route = '/users/%s/roles/OS-KSADM/%s'
        return self._update(route % (user_id, role_id), None, 'roles')