from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def EncodeUpdatedTriggerSubstitutions(old_substitutions, substitutions, messages):
    """Encodes the trigger substitutions for the update command.

  Args:
    old_substitutions: The existing substitutions to be updated.
    substitutions: The substitutions to be added to the existing substitutions.
    messages: A Cloud Build messages module.

  Returns:
    The updated trigger substitutions.
  """
    if not substitutions:
        return old_substitutions
    substitution_map = {}
    if old_substitutions:
        for sub in old_substitutions.additionalProperties:
            substitution_map[sub.key] = sub.value
    for key, value in six.iteritems(substitutions):
        substitution_map[key] = value
    updated_substitutions = []
    for key, value in sorted(substitution_map.items()):
        updated_substitutions.append(messages.BuildTrigger.SubstitutionsValue.AdditionalProperty(key=key, value=value))
    return messages.BuildTrigger.SubstitutionsValue(additionalProperties=updated_substitutions)