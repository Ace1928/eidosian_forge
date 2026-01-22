from keystoneclient import base
from keystoneclient.i18n import _
from keystoneclient.v3 import endpoints
from keystoneclient.v3 import policies
def _act_on_policy_association_for_region_and_service(self, policy, region, service, action):
    if not (policy and region and service):
        raise ValueError(_('policy, region and service are required'))
    policy_id = base.getid(policy)
    region_id = base.getid(region)
    service_id = base.getid(service)
    url = '/policies/%(policy_id)s/%(ext_name)s/services/%(service_id)s/regions/%(region_id)s' % {'policy_id': policy_id, 'ext_name': self.OS_EP_POLICY_EXT, 'service_id': service_id, 'region_id': region_id}
    return action(url=url)