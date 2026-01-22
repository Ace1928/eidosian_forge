from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ClearSingleEndpointAttr(patch_request, endpoint_type, endpoint_name):
    """Checks if given endpoint can be removed from Connectivity Test and removes it."""
    test = patch_request.connectivityTest
    endpoint = getattr(test, endpoint_type)
    endpoint_fields = {'instance', 'ipAddress', 'gkeMasterCluster', 'cloudSqlInstance', 'cloudFunction', 'appEngineVersion', 'cloudRunRevision', 'forwardingRule'}
    non_empty_endpoint_fields = 0
    for field in endpoint_fields:
        if getattr(endpoint, field, None):
            non_empty_endpoint_fields += 1
    if non_empty_endpoint_fields > 1 or not getattr(endpoint, endpoint_name, None):
        ClearEndpointValue(endpoint, endpoint_name)
        setattr(test, endpoint_type, endpoint)
        patch_request.connectivityTest = test
        return AddFieldToUpdateMask(endpoint_type + '.' + endpoint_name, patch_request)
    else:
        endpoints = ['instance', 'ip-address', 'gke-master-cluster', 'cloud-sql-instance']
        if endpoint_type == 'source':
            endpoints.extend(['cloud-function', 'app-engine-version', 'cloud-run-revision'])
        if endpoint_type == 'destination':
            endpoints.extend(['forwarding-rule'])
        raise InvalidInputError(GetClearSingleEndpointAttrErrorMsg(endpoints, endpoint_type))