from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateSchemaResourceArg(verb, positional=True, plural=False, required=True, flag_name='schema'):
    """Create a resource argument for a Cloud Pub/Sub Schema.

  Args:
    verb: str, the verb to describe the resource, such as 'to update'.
    positional: bool, if True, means that the schema ID is a positional rather
      than a flag. If not positional, this also creates a '--schema-project'
      flag as schemas and topics do not need to be in the same project.
    plural: bool, if True, use a resource argument that returns a list.
    required: bool, if True, schema resource arg will be required.
    flag_name: str, name of the schema resource arg (singular).

  Returns:
    the PresentationSpec for the resource argument.
  """
    if positional:
        name = flag_name
        flag_name_overrides = {}
    else:
        name = '--' + flag_name if not plural else '--' + flag_name + 's'
        flag_name_overrides = {'project': '--' + flag_name + '-project'}
    help_stem = 'Name of the schema'
    if plural:
        help_stem = 'One or more schemas'
    return presentation_specs.ResourcePresentationSpec(name, GetSchemaResourceSpec(flag_name), '{} {}'.format(help_stem, verb), required=required, flag_name_overrides=flag_name_overrides, plural=plural, prefixes=True)