from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _HasSteps(intoto):
    """Check whether a build provenance contains build steps.

  Args:
    intoto: intoto statement in build occurrence.

  Returns:
    A boolean value indicating whether intoto contains build steps.
  """
    if intoto and hasattr(intoto, 'slsaProvenance') and hasattr(intoto.slsaProvenance, 'recipe') and hasattr(intoto.slsaProvenance.recipe, 'arguments') and hasattr(intoto.slsaProvenance.recipe.arguments, 'additionalProperties'):
        properties = intoto.slsaProvenance.recipe.arguments.additionalProperties
        return any((p.key == 'steps' and p.value for p in properties))
    return False