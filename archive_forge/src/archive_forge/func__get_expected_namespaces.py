import http.client as http
from oslo_serialization import jsonutils
import requests
from glance.tests.functional.v2 import metadef_base
def _get_expected_namespaces(tenant):
    expected_namespaces = []
    for x in tenant_namespaces[tenant]:
        expected_namespaces.append(x['namespace'])
    if tenant == self.tenant1:
        expected_namespaces.append(tenant_namespaces[self.tenant2][0]['namespace'])
    else:
        expected_namespaces.append(tenant_namespaces[self.tenant1][0]['namespace'])
    return expected_namespaces