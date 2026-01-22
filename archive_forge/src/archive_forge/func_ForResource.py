from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
@classmethod
def ForResource(cls, name, resource_spec, group_help, required=False, hidden=False, flag_name_overrides=None, plural=False, prefixes=False, group=None, command_level_fallthroughs=None):
    """Constructs a ConceptParser for a single resource argument.

    Automatically sets prefixes to False.

    Args:
      name: str, the name of the main arg for the resource.
      resource_spec: googlecloudsdk.calliope.concepts.ResourceSpec, The spec
        that specifies the resource.
      group_help: str, the help text for the entire arg group.
      required: bool, whether the main argument should be required for the
        command.
      hidden: bool, whether or not the resource is hidden.
      flag_name_overrides: {str: str}, dict of attribute names to the desired
        flag name. To remove a flag altogether, use '' as its rename value.
      plural: bool, True if the resource will be parsed as a list, False
        otherwise.
      prefixes: bool, True if flag names will be prefixed with the resource
        name, False otherwise. Should be False for all typical use cases.
      group: the parser or subparser for a Calliope command that the resource
        arguments should be added to. If not provided, will be added to the main
        parser.
      command_level_fallthroughs: a map of attribute names to lists of command-
        specific fallthroughs. These will be prioritized over the default
        fallthroughs for the attribute.

    Returns:
      (googlecloudsdk.calliope.concepts.concept_parsers.ConceptParser) The fully
        initialized ConceptParser.
    """
    presentation_spec = presentation_specs.ResourcePresentationSpec(name, resource_spec, group_help, required=required, flag_name_overrides=flag_name_overrides or {}, plural=plural, prefixes=prefixes, group=group, hidden=hidden)
    fallthroughs_map = {}
    UpdateFallthroughsMap(fallthroughs_map, name, command_level_fallthroughs)
    for attribute_name, fallthroughs in six.iteritems(command_level_fallthroughs or {}):
        key = '{}.{}'.format(presentation_spec.name, attribute_name)
        fallthroughs_map[key] = fallthroughs
    return cls([presentation_spec], fallthroughs_map)