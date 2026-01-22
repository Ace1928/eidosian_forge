from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def _GetCertificateResourcePresentationSpec(flag, noun, verb, required=True, plural=False, group=None, with_location=True):
    """Returns ResourcePresentationSpec for certificate resource.

  Args:
    flag: str, the flag name.
    noun: str, the resource; default: 'The certificate map'.
    verb: str, the verb to describe the resource, such as 'to update'.
    required: bool, if False, means that map ID is optional.
    plural: bool.
    group: args group.
    with_location: bool, if False, means that location flag is hidden.

  Returns:
    presentation_specs.ResourcePresentationSpec.
  """
    flag_name_overrides = {}
    if not with_location:
        flag_name_overrides['location'] = ''
    return presentation_specs.ResourcePresentationSpec(flag, GetCertificateResourceSpec(), '{} {}.'.format(noun, verb), required=required, plural=plural, group=group, flag_name_overrides=flag_name_overrides)