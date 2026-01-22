from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def AddDefaultReplicaCount(unused_instance_ref, args, post_request):
    """Hook to update default replica count."""
    if args.IsSpecified('replica_count'):
        return post_request
    if args.read_replicas_mode == 'read-replicas-enabled':
        post_request.instance.replicaCount = 2
    return post_request