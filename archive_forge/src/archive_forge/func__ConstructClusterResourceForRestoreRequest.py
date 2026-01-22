from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.core import properties
def _ConstructClusterResourceForRestoreRequest(alloydb_messages, args):
    """Returns the cluster resource for restore request."""
    cluster_resource = alloydb_messages.Cluster()
    cluster_resource.network = args.network
    kms_key = flags.GetAndValidateKmsKeyName(args)
    if kms_key:
        encryption_config = alloydb_messages.EncryptionConfig()
        encryption_config.kmsKeyName = kms_key
        cluster_resource.encryptionConfig = encryption_config
    if args.allocated_ip_range_name:
        cluster_resource.networkConfig = alloydb_messages.NetworkConfig(allocatedIpRange=args.allocated_ip_range_name)
    if args.enable_private_service_connect:
        cluster_resource.pscConfig = alloydb_messages.PscConfig(pscEnabled=True)
    return cluster_resource