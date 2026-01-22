from openstack import resource
from openstack import utils
class ProfileType(resource.Resource):
    resource_key = 'profile_type'
    resources_key = 'profile_types'
    base_path = '/profile-types'
    allow_list = True
    allow_fetch = True
    name = resource.Body('name', alternate_id=True)
    schema = resource.Body('schema')
    support_status = resource.Body('support_status')

    def type_ops(self, session):
        url = utils.urljoin(self.base_path, self.id, 'ops')
        resp = session.get(url)
        return resp.json()