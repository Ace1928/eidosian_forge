from openstack.common import tag
from openstack import resource
from openstack import utils
def unassign_role_from_user(self, session, user, role):
    """Unassigns a role from a user on a project"""
    url = utils.urljoin(self.base_path, self.id, 'users', user.id, 'roles', role.id)
    resp = session.delete(url)
    if resp.status_code == 204:
        return True
    return False