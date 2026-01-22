from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def _TransformLocations(r):
    if 'replication' not in r:
        return 'ERROR'
    if 'automatic' in r['replication']:
        return '-'
    if 'userManaged' in r['replication'] and 'replicas' in r['replication']['userManaged']:
        locations = []
        for replica in r['replication']['userManaged']['replicas']:
            locations.append(replica['location'])
        return ','.join(locations)
    return 'ERROR'