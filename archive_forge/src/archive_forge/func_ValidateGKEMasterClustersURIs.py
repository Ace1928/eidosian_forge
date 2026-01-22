from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
def ValidateGKEMasterClustersURIs(unused_ref, args, request):
    """Checks if all provided GKE Master Clusters URIs are in correct format."""
    flags = ['source_gke_master_cluster', 'destination_gke_master_cluster']
    instance_pattern = re.compile('projects/(?:[a-z][a-z0-9-\\.:]*[a-z0-9])/(zones|locations)/[-\\w]+/clusters/[-\\w]+')
    for flag in flags:
        if args.IsSpecified(flag):
            cluster = getattr(args, flag)
            if not instance_pattern.match(cluster):
                raise InvalidInputError('Invalid value for flag {}: {}\nExpected Google Kubernetes Engine master cluster in the following format:\n  projects/my-project/location/location/clusters/my-cluster'.format(flag, cluster))
    return request