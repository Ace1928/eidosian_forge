from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
@classmethod
def FromVersionResource(cls, version, service):
    """Convert appengine_<API-version>_messages.Version into wrapped Version."""
    project, service_id, _ = re.match(cls._VERSION_NAME_PATTERN, version.name).groups()
    traffic_split = service and service.split.get(version.id, 0.0)
    last_deployed = None
    try:
        if version.createTime:
            last_deployed_dt = times.ParseDateTime(version.createTime).replace(microsecond=0)
            last_deployed = times.LocalizeDateTime(last_deployed_dt)
    except ValueError:
        pass
    if version.env == 'flexible':
        environment = env.FLEX
    elif version.vm:
        environment = env.MANAGED_VMS
    else:
        environment = env.STANDARD
    return cls(project, service_id, version.id, traffic_split=traffic_split, last_deployed_time=last_deployed, environment=environment, version_resource=version)