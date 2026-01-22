from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
from six.moves import map  # pylint: disable=redefined-builtin
def FilterInstances(instances, service=None, version=None, instance=None):
    """Filter a list of App Engine instances.

  Args:
    instances: list of AppEngineInstance, all App Engine instances
    service: str, the name of the service to filter by or None to match all
      services
    version: str, the name of the version to filter by or None to match all
      versions
    instance: str, the instance id to filter by or None to match all versions.

  Returns:
    list of instances matching the given filters
  """
    matching_instances = []
    for provided_instance in instances:
        if (not service or provided_instance.service == service) and (not version or provided_instance.version == version) and (not instance or provided_instance.id == instance):
            matching_instances.append(provided_instance)
    return matching_instances