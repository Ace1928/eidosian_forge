from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def ListMemberships(project):
    """List hte memberships from a given project.

  Args:
    project: project that the memberships are in.

  Returns:
    The memberships registered to the fleet hosted by the given project.

  Raises:
    Error: The error occured when it failed to list memberships.
  """
    args = ['container', 'fleet', 'memberships', 'list', '--format', 'json(name)', '--project', project]
    output, err = _RunGcloud(args)
    if err:
        raise exceptions.ConfigSyncError('Error listing memberships: {}'.format(err))
    json_output = json.loads(output)
    memberships = [m['name'] for m in json_output]
    return memberships