from openstack import resource
class PolicyValidate(Policy):
    base_path = '/policies/validate'
    allow_list = False
    allow_fetch = False
    allow_create = True
    allow_delete = False
    allow_commit = False
    commit_method = 'PUT'