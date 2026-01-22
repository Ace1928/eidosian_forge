from openstack.common import tag
from openstack import resource
class LoadBalancerFailover(resource.Resource):
    base_path = '/lbaas/loadbalancers/%(lb_id)s/failover'
    allow_create = False
    allow_fetch = False
    allow_commit = True
    allow_delete = False
    allow_list = False
    allow_empty_commit = True
    requires_id = False
    lb_id = resource.URI('lb_id')

    def commit(self, session, base_path=None):
        return super(LoadBalancerFailover, self).commit(session, base_path=base_path, has_body=False)