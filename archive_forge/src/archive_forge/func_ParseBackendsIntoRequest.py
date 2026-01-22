from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ParseBackendsIntoRequest(job_ref, request):
    """Generate the long backend name of Dataproc Metastore federation requests.

  Args:
    job_ref: A resource ref to the parsed Federation resource.
    request: The request for the API call.

  Returns:
    Modified request for the API call.
  """
    for prop in request.federation.backendMetastores.additionalProperties:
        prop.value.name = prop.value.name.format(job_ref.Parent().RelativeName())
    return request