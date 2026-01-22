from openstack import resource
class FlavorProfile(resource.Resource):
    resource_key = 'flavorprofile'
    resources_key = 'flavorprofiles'
    base_path = '/lbaas/flavorprofiles'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('id', 'name', 'provider_name', 'flavor_data')
    id = resource.Body('id')
    name = resource.Body('name')
    provider_name = resource.Body('provider_name')
    flavor_data = resource.Body('flavor_data')