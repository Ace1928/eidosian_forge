from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreatePatchJobResourceArg(verb, plural=False):
    """Creates a resource argument for a OS Config patch job.

  Args:
    verb: str, The verb to describe the resource, such as 'to describe'.
    plural: bool, If True, use a resource argument that returns a list.

  Returns:
    PresentationSpec for the resource argument.
  """
    noun = 'Patch job' + ('s' if plural else '')
    return presentation_specs.ResourcePresentationSpec('patch_job', GetPatchJobResourceSpec(), '{} {}'.format(noun, verb), required=True, plural=plural, prefixes=False)