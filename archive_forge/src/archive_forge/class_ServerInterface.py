from openstack import resource
class ServerInterface(resource.Resource):
    resource_key = 'interfaceAttachment'
    resources_key = 'interfaceAttachments'
    base_path = '/servers/%(server_id)s/os-interface'
    allow_create = True
    allow_fetch = True
    allow_commit = False
    allow_delete = True
    allow_list = True
    fixed_ips = resource.Body('fixed_ips')
    mac_addr = resource.Body('mac_addr')
    net_id = resource.Body('net_id')
    port_id = resource.Body('port_id', alternate_id=True)
    port_state = resource.Body('port_state')
    server_id = resource.URI('server_id')
    tag = resource.Body('tag')
    _max_microversion = '2.70'