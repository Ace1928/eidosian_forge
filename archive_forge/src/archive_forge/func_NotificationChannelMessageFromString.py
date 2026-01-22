from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def NotificationChannelMessageFromString(alert_manager_yaml, messages):
    """Populates Alert Policies translated from Prometheus alert rules.

  Args:
    alert_manager_yaml: Opened object of the Prometheus YAML file provided.
    messages: Object containing information about all message types allowed.

  Raises:
    YamlOrJsonLoadError: If the YAML file cannot be loaded.

  Returns:
     The Alert Policies corresponding to the Prometheus rules YAML file
     provided.
  """
    try:
        contents = yaml.load(alert_manager_yaml)
    except Exception as exc:
        raise YamlOrJsonLoadError('Could not parse YAML: {0}'.format(exc))
    channels = []
    for receiver_config in contents.get('receivers'):
        channels += BuildChannelsFromPrometheusReceivers(receiver_config, messages)
    return channels