from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import ipaddr
def _FormatResourceRecordSet(rrdatas_or_routing_policy):
    """Format rrset based on rrdatas or routing policy type."""
    if 'wrr' in rrdatas_or_routing_policy:
        items = []
        for item in rrdatas_or_routing_policy['wrr']['items']:
            items.append('{}: {}'.format(item['weight'], _FormatRrdata(item)))
        return '; '.join(items)
    elif 'geo' in rrdatas_or_routing_policy:
        items = []
        for item in rrdatas_or_routing_policy['geo']['items']:
            items.append('{}: {}'.format(item['location'], _FormatRrdata(item)))
        return '; '.join(items)
    elif 'primaryBackup' in rrdatas_or_routing_policy:
        items = []
        for item in rrdatas_or_routing_policy['primaryBackup']['backupGeoTargets']['items']:
            items.append('{}: {}'.format(item['location'], _FormatRrdata(item)))
        backup = ';'.join(items)
        primary = ','.join(('"{}"'.format(_FormatHealthCheckTarget(target)) for target in rrdatas_or_routing_policy['primaryBackup']['primaryTargets']['internalLoadBalancers']))
        return 'Primary: {} Backup: {}'.format(primary, backup)
    else:
        return ','.join(rrdatas_or_routing_policy)