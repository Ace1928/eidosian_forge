from openstack import resource
class AddressScope(resource.Resource):
    """Address scope extension."""
    resource_key = 'address_scope'
    resources_key = 'address_scopes'
    base_path = '/address-scopes'
    _allow_unknown_attrs_in_body = True
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'ip_version', 'project_id', 'sort_key', 'sort_dir', is_shared='shared')
    name = resource.Body('name')
    project_id = resource.Body('project_id', alias='tenant_id')
    tenant_id = resource.Body('tenant_id', deprecated=True)
    ip_version = resource.Body('ip_version', type=int)
    is_shared = resource.Body('shared', type=bool)