from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetLookupEntryArg():
    """Returns the argument for looking up an entry."""
    help_text = 'The name of the target resource whose entry to update. This can be either the\nGoogle Cloud Platform resource name or the SQL name of a Google Cloud Platform\nresource. This flag allows one to update the entry corresponding to the lookup\nof the given resource, without needing to specify the entry directly.'
    return base.Argument('--lookup-entry', metavar='RESOURCE', help=help_text)