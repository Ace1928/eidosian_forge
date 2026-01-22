from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ClearEndpointAttrsBeta(unused_ref, args, patch_request):
    """Handles clear_source_* and clear_destination_* flags."""
    flags_and_endpoints = [('clear_source_instance', 'source', 'instance'), ('clear_source_ip_address', 'source', 'ipAddress'), ('clear_source_gke_master_cluster', 'source', 'gkeMasterCluster'), ('clear_source_cloud_sql_instance', 'source', 'cloudSqlInstance'), ('clear_source_cloud_function', 'source', 'cloudFunction'), ('clear_source_app_engine_version', 'source', 'appEngineVersion'), ('clear_source_cloud_run_revision', 'source', 'cloudRunRevision'), ('clear_destination_instance', 'destination', 'instance'), ('clear_destination_ip_address', 'destination', 'ipAddress'), ('clear_destination_gke_master_cluster', 'destination', 'gkeMasterCluster'), ('clear_destination_cloud_sql_instance', 'destination', 'cloudSqlInstance'), ('clear_destination_forwarding_rule', 'destination', 'forwardingRule')]
    for flag, endpoint_type, endpoint_name in flags_and_endpoints:
        if args.IsSpecified(flag):
            patch_request = ClearSingleEndpointAttrBeta(patch_request, endpoint_type, endpoint_name)
    return patch_request