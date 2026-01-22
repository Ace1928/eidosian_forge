from openstack import resource
class QoSBandwidthLimitRule(resource.Resource):
    resource_key = 'bandwidth_limit_rule'
    resources_key = 'bandwidth_limit_rules'
    base_path = '/qos/policies/%(qos_policy_id)s/bandwidth_limit_rules'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    qos_policy_id = resource.URI('qos_policy_id')
    max_kbps = resource.Body('max_kbps')
    max_burst_kbps = resource.Body('max_burst_kbps')
    direction = resource.Body('direction')