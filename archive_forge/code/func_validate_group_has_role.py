from openstack.common import tag
from openstack import resource
from openstack import utils
def validate_group_has_role(self, session, group, role):
    """Validates that a group has a role on a project"""
    url = utils.urljoin(self.base_path, self.id, 'groups', group.id, 'roles', role.id)
    resp = session.head(url)
    if resp.status_code == 204:
        return True
    return False