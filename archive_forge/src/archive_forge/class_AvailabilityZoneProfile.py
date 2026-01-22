from openstack import resource
class AvailabilityZoneProfile(resource.Resource):
    resource_key = 'availability_zone_profile'
    resources_key = 'availability_zone_profiles'
    base_path = '/lbaas/availabilityzoneprofiles'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('id', 'name', 'provider_name', 'availability_zone_data')
    id = resource.Body('id')
    name = resource.Body('name')
    provider_name = resource.Body('provider_name')
    availability_zone_data = resource.Body('availability_zone_data')