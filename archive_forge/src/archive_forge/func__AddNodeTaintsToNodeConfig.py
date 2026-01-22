from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _AddNodeTaintsToNodeConfig(self, node_config, options):
    """Add nodeTaints to nodeConfig."""
    if options.node_taints is None:
        return
    taints = []
    effect_enum = self.messages.NodeTaint.EffectValueValuesEnum
    for key, value in sorted(six.iteritems(options.node_taints)):
        strs = value.split(':')
        if len(strs) != 2:
            raise util.Error(NODE_TAINT_INCORRECT_FORMAT_ERROR_MSG.format(key=key, value=value))
        value = strs[0]
        taint_effect = strs[1]
        if taint_effect == 'NoSchedule':
            effect = effect_enum.NO_SCHEDULE
        elif taint_effect == 'PreferNoSchedule':
            effect = effect_enum.PREFER_NO_SCHEDULE
        elif taint_effect == 'NoExecute':
            effect = effect_enum.NO_EXECUTE
        else:
            raise util.Error(NODE_TAINT_INCORRECT_EFFECT_ERROR_MSG.format(effect=strs[1]))
        taints.append(self.messages.NodeTaint(key=key, value=value, effect=effect))
    node_config.taints = taints