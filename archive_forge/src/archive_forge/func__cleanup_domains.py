from openstack import exceptions
from openstack.tests.functional import base
def _cleanup_domains(self):
    exception_list = list()
    for domain in self.operator_cloud.list_domains():
        if domain['name'].startswith(self.domain_prefix):
            try:
                self.operator_cloud.delete_domain(domain['id'])
            except Exception as e:
                exception_list.append(str(e))
                continue
    if exception_list:
        raise exceptions.SDKException('\n'.join(exception_list))