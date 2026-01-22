from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def PropertySpecTransform(property_spec):
    """Mutates the given property spec from Tekton to GCB format.

  Args:
    property_spec: A Tekton-compliant property spec.
  """
    _ConvertToUpperCase(property_spec, 'type')