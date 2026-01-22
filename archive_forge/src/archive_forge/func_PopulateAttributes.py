from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
def PopulateAttributes(args, release_track=base.ReleaseTrack.ALPHA):
    """Populate attirbutes from args."""
    attributes = GetMessagesModule(release_track).Attributes()
    if args.environment_type:
        attributes.environment = GetMessagesModule(release_track).Environment(type=GetMessagesModule(release_track).Environment.TypeValueValuesEnum(args.environment_type))
    if args.criticality_type:
        attributes.criticality = GetMessagesModule(release_track).Criticality(type=GetMessagesModule(release_track).Criticality.TypeValueValuesEnum(args.criticality_type))
    for b_owner in args.business_owners or []:
        business_owner = GetMessagesModule(release_track).ContactInfo()
        business_owner.email = b_owner.get('email', None)
        if b_owner.get('display-name', None):
            business_owner.displayName = b_owner.get('display-name', None)
        if release_track == base.ReleaseTrack.ALPHA:
            if b_owner.get('channel-uri', None):
                business_owner.channel = GetMessagesModule(release_track).Channel(uri=b_owner.get('channel-uri'))
        attributes.businessOwners.append(business_owner)
    for d_owner in args.developer_owners or []:
        developer_owner = GetMessagesModule(release_track).ContactInfo()
        developer_owner.email = d_owner.get('email', None)
        if d_owner.get('display-name', None):
            developer_owner.displayName = d_owner.get('display-name', None)
        if release_track == base.ReleaseTrack.ALPHA:
            if d_owner.get('channel-uri', None):
                developer_owner.channel = GetMessagesModule(release_track).Channel(uri=d_owner.get('channel-uri'))
        attributes.developerOwners.append(developer_owner)
    for o_owner in args.operator_owners or []:
        operator_owner = GetMessagesModule(release_track).ContactInfo()
        operator_owner.email = o_owner.get('email', None)
        if o_owner.get('display-name'):
            operator_owner.displayName = o_owner.get('display-name')
        if release_track == base.ReleaseTrack.ALPHA:
            if o_owner.get('channel-uri'):
                operator_owner.channel = GetMessagesModule(release_track).Channel(uri=o_owner.get('channel-uri'))
        attributes.operatorOwners.append(operator_owner)
    return attributes