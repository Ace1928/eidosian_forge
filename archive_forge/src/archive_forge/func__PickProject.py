from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import usage_text
from googlecloudsdk.command_lib import init_util
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.credentials import store as c_store
from googlecloudsdk.core.diagnostics import network_diagnostics
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _PickProject(self, preselected=None):
    """Allows user to select a project.

    Args:
      preselected: str, use this value if not None

    Returns:
      str, project_id or None if was not selected.
    """
    project_id = init_util.PickProject(preselected=preselected)
    if project_id is not None:
        properties.PersistProperty(properties.VALUES.core.project, project_id)
        log.status.write('Your current project has been set to: [{0}].\n\n'.format(project_id))
    return project_id