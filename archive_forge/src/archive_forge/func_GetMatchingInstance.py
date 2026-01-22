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
def GetMatchingInstance(instances, service=None, version=None, instance=None):
    """Return exactly one matching instance.

  If instance is given, filter down based on the given criteria (service,
  version, instance) and return the matching instance (it is an error unless
  exactly one instance matches).

  Otherwise, prompt the user to select the instance interactively.

  Args:
    instances: list of AppEngineInstance, all instances to select from
    service: str, a service to filter by or None to include all services
    version: str, a version to filter by or None to include all versions
    instance: str, an instance ID to filter by. If not given, the instance will
      be selected interactively.

  Returns:
    AppEngineInstance, an instance from the given list.

  Raises:
    InvalidInstanceSpecificationError: if no matching instances or more than one
      matching instance were found.
  """
    if not instance:
        return SelectInstanceInteractive(instances, service=service, version=version)
    matching = FilterInstances(instances, service, version, instance)
    if len(matching) > 1:
        raise InvalidInstanceSpecificationError('More than one instance matches the given specification.\n\nMatching instances: {0}'.format(list(sorted(map(str, matching)))))
    elif not matching:
        raise InvalidInstanceSpecificationError('No instances match the given specification.\n\nAll instances: {0}'.format(list(sorted(map(str, instances)))))
    return matching[0]