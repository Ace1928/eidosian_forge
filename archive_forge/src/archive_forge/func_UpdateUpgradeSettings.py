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
def UpdateUpgradeSettings(self, node_pool_ref, options, pool=None):
    """Updates node pool's upgrade setting."""
    if pool is None:
        pool = self.GetNodePool(node_pool_ref)
    if options.enable_surge_upgrade and options.enable_blue_green_upgrade:
        raise util.Error('UpgradeSettings must contain only one of: --enable-surge-upgrade, --enable-blue-green-upgrade')
    upgrade_settings = pool.upgradeSettings
    if upgrade_settings is None:
        upgrade_settings = self.messages.UpgradeSettings()
    if options.max_surge_upgrade is not None:
        upgrade_settings.maxSurge = options.max_surge_upgrade
    if options.max_unavailable_upgrade is not None:
        upgrade_settings.maxUnavailable = options.max_unavailable_upgrade
    if options.enable_surge_upgrade:
        upgrade_settings.strategy = self.messages.UpgradeSettings.StrategyValueValuesEnum.SURGE
    if options.enable_blue_green_upgrade:
        upgrade_settings.strategy = self.messages.UpgradeSettings.StrategyValueValuesEnum.BLUE_GREEN
    if options.standard_rollout_policy is not None or options.node_pool_soak_duration is not None:
        upgrade_settings.blueGreenSettings = self.UpdateBlueGreenSettings(upgrade_settings, options)
    if options.autoscaled_rollout_policy:
        upgrade_settings.blueGreenSettings = self.UpdateBlueGreenSettings(upgrade_settings, options)
    return upgrade_settings