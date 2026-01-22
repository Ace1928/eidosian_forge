from openstack import resource
class DomainConfigLDAP(resource.Resource):
    user_tree_dn = resource.Body('user_tree_dn')
    url = resource.Body('url')