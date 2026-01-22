from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
def _ParseRepairActionFunc(repair_action=None):
    return arg_utils.ChoiceToEnum(repair_action, dataproc.messages.NodePool.RepairActionValueValuesEnum, item_type='REPAIR_ACTION', valid_choices=cls._GetValidRepairActionChoices(dataproc))