import abc
from keystoneauth1.identity.v3 import base
from keystoneauth1.identity.v3 import token
def _get_scoping_data(self):
    return {'trust_id': self.trust_id, 'domain_id': self.domain_id, 'domain_name': self.domain_name, 'project_id': self.project_id, 'project_name': self.project_name, 'project_domain_id': self.project_domain_id, 'project_domain_name': self.project_domain_name}