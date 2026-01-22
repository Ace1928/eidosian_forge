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
def WorkerPoolShortName(resource_name):
    """Get the name part of a worker pool's full resource name.

  For example, "projects/abc/locations/def/workerPools/ghi" returns "ghi".

  Args:
    resource_name: A worker pool's full resource name.

  Raises:
    ValueError: If the full resource name was not well-formatted.

  Returns:
    The worker pool's short name.
  """
    match = re.search(WORKERPOOL_NAME_SELECTOR, resource_name)
    if match:
        return match.group(1)
    raise ValueError('The worker pool resource name must match "%s"' % (WORKERPOOL_NAME_MATCHER,))