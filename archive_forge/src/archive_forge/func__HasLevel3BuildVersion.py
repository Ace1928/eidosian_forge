from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.services import enable_api
import six
def _HasLevel3BuildVersion(intoto):
    """Check whether a build provenance contains level 3 build version.

  Args:
    intoto: intoto statement in build occurrence.

  Returns:
    A boolean value indicating whether intoto contains level 3 build version.
  """
    if intoto and hasattr(intoto, 'slsaProvenance') and hasattr(intoto.slsaProvenance, 'builder') and hasattr(intoto.slsaProvenance.builder, 'id') and intoto.slsaProvenance.builder.id:
        [uri, version] = intoto.slsaProvenance.builder.id.split('@v')
        if uri == 'https://cloudbuild.googleapis.com/GoogleHostedWorker' and version:
            [major_version, minor_version] = version.split('.')
            return int(major_version) > 0 or int(minor_version) >= 3
    return False