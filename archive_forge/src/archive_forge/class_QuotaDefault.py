from openstack import resource
class QuotaDefault(Quota):
    base_path = '/quotas/%(project)s/default'
    allow_retrieve = True
    allow_commit = False
    allow_delete = False
    allow_list = False
    project = resource.URI('project')