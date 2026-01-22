from openstack import resource
class ServerUsage(resource.Resource):
    resource_key = None
    resources_key = None
    allow_create = False
    allow_fetch = False
    allow_delete = False
    allow_list = False
    allow_commit = False
    hours = resource.Body('hours')
    flavor = resource.Body('flavor')
    instance_id = resource.Body('instance_id')
    name = resource.Body('name')
    project_id = resource.Body('tenant_id')
    memory_mb = resource.Body('memory_mb')
    local_gb = resource.Body('local_gb')
    vcpus = resource.Body('vcpus')
    started_at = resource.Body('started_at')
    ended_at = resource.Body('ended_at')
    state = resource.Body('state')
    uptime = resource.Body('uptime')