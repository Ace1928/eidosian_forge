from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
from googlecloudsdk.core.docker import client_lib as client_utils
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import semver
import six
def BuildOrderedCredentialHelperRegistries(registries):
    """Returns dict of gcloud helper mappings for the supplied repositories.

  Returns ordered dict of Docker registry to gcloud helper mappings for the
  supplied list of registries.

  Ensures that the order in which credential helper registry entries are
  processed is consistent.

  Args:
      registries: list, the registries to create the mappings for.

  Returns:
   OrderedDict of Docker registry to gcloud helper mappings.
  """
    return collections.OrderedDict([(registry, 'gcloud') for registry in registries])