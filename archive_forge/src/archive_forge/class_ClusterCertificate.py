from openstack import resource
class ClusterCertificate(resource.Resource):
    base_path = '/certificates'
    allow_create = True
    allow_list = False
    allow_fetch = True
    bay_uuid = resource.Body('bay_uuid')
    cluster_uuid = resource.Body('cluster_uuid', alternate_id=True)
    csr = resource.Body('csr')
    pem = resource.Body('pem')