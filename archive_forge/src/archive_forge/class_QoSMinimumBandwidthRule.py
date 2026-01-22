from openstack import resource
class QoSMinimumBandwidthRule(resource.Resource):
    resource_key = 'minimum_bandwidth_rule'
    resources_key = 'minimum_bandwidth_rules'
    base_path = '/qos/policies/%(qos_policy_id)s/minimum_bandwidth_rules'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    qos_policy_id = resource.URI('qos_policy_id')
    min_kbps = resource.Body('min_kbps')
    direction = resource.Body('direction')