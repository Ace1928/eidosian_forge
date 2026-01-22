from openstack import resource
class QoSDSCPMarkingRule(resource.Resource):
    resource_key = 'dscp_marking_rule'
    resources_key = 'dscp_marking_rules'
    base_path = '/qos/policies/%(qos_policy_id)s/dscp_marking_rules'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    qos_policy_id = resource.URI('qos_policy_id')
    dscp_mark = resource.Body('dscp_mark')