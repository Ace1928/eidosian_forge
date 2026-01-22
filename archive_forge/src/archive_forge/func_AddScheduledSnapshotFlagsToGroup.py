from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddScheduledSnapshotFlagsToGroup(update_type_group):
    """Adds flags related to scheduled snapshot.

  Args:
    update_type_group: argument group, the group to which flags should be added.
  """
    update_group = update_type_group.add_argument_group(SCHEDULED_SNAPSHOTS_UPDATE_GROUP_DESCRIPTION, mutex=True)
    DISABLE_SCHEDULED_SNAPSHOT_CREATION.AddToParser(update_group)
    scheduled_snapshots_params_group = update_group.add_argument_group(SCHEDULED_SNAPSHOTS_GROUP_DESCRIPTION)
    ENABLE_SCHEDULED_SNAPSHOT_CREATION.AddToParser(scheduled_snapshots_params_group)
    SNAPSHOT_LOCATION.AddToParser(scheduled_snapshots_params_group)
    SNAPSHOT_CREATION_SCHEDULE.AddToParser(scheduled_snapshots_params_group)
    SNAPSHOT_SCHEDULE_TIMEZONE.AddToParser(scheduled_snapshots_params_group)