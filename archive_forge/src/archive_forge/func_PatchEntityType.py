from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
import six
def PatchEntityType(unused_instance_ref, args, update_request):
    """Update entities based on clear/remove/add-entities flags."""
    entities = update_request.googleCloudDialogflowV2EntityType.entities
    if args.IsSpecified('clear_entities'):
        entities = []
    if args.IsSpecified('remove_entities'):
        to_remove = set(args.remove_entities or [])
        entities = [e for e in entities if e.value not in to_remove]
    if args.IsSpecified('add_entities'):
        entities += args.add_entities
    update_request.googleCloudDialogflowV2EntityType.entities = entities
    return update_request