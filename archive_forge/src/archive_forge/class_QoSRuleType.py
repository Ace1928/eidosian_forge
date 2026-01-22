from openstack import resource
class QoSRuleType(resource.Resource):
    resource_key = 'rule_type'
    resources_key = 'rule_types'
    base_path = '/qos/rule-types'
    _allow_unknown_attrs_in_body = True
    allow_create = False
    allow_fetch = True
    allow_commit = False
    allow_delete = False
    allow_list = True
    _query_mapping = resource.QueryParameters('type', 'drivers', 'all_rules', 'all_supported')
    type = resource.Body('type', alternate_id=True)
    drivers = resource.Body('drivers')