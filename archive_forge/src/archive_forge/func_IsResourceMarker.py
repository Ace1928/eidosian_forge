from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_property
def IsResourceMarker(resource):
    """Returns True if resource is a _ResourceMarker."""
    return isinstance(resource, _ResourceMarker)