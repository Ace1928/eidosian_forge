from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.compute.instances.ops_agents import exceptions
from googlecloudsdk.api_lib.compute.instances.ops_agents import ops_agents_policy as agent_policy
from googlecloudsdk.core import log
Validates the combination of the OS short name and version is supported.

  This validation happens after the arg parsing stage. At this point, we can
  assume that the field is an OpsAgentPolicy.Assignment.OsType object. Also the
  OS short name has been already validated at the arg parsing stage.

  Args:
    short_name: str. The OS short name to filter instances by.
    version: str. The OS version to filter instances by.

  Returns:
    An empty list if the validation passes. A singleton list with the following
    error if the validation fails.
    * OsTypeNotSupportedError:
      The combination of the OS short name and version is not supported.
  