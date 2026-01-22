from openstack import resource
class PolicyType(resource.Resource):
    resource_key = 'policy_type'
    resources_key = 'policy_types'
    base_path = '/policy-types'
    allow_list = True
    allow_fetch = True
    name = resource.Body('name', alternate_id=True)
    schema = resource.Body('schema')
    support_status = resource.Body('support_status')