from openstack import resource
class ClusterAttr(resource.Resource):
    resources_key = 'cluster_attributes'
    base_path = '/clusters/%(cluster_id)s/attrs/%(path)s'
    allow_list = True
    cluster_id = resource.URI('cluster_id')
    path = resource.URI('path')
    node_id = resource.Body('id')
    attr_value = resource.Body('value')