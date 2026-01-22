from openstack import resource
class IdentityProvider(resource.Resource):
    resource_key = 'identity_provider'
    resources_key = 'identity_providers'
    base_path = '/OS-FEDERATION/identity_providers'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    create_method = 'PUT'
    create_exclude_id_from_body = True
    commit_method = 'PATCH'
    _query_mapping = resource.QueryParameters('id', is_enabled='enabled')
    domain_id = resource.Body('domain_id')
    description = resource.Body('description')
    is_enabled = resource.Body('enabled', type=bool)
    remote_ids = resource.Body('remote_ids', type=list)
    name = resource.Body('id')