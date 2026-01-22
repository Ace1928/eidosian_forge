from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def ValidateAutoprovisioningConfigFile(nap_config_file):
    """Load and Validate Autoprovisioning configuration from YAML/JSON file.

  Args:
    nap_config_file: The YAML/JSON string that contains sysctl and kubelet
      options.

  Raises:
    Error: when there's any errors on parsing the YAML/JSON system config
    or wrong name are present.
  """
    try:
        nap_config = yaml.load(nap_config_file)
    except yaml.YAMLParseError as e:
        raise AutoprovisioningConfigError('autoprovisioning config file is not valid YAML/JSON: {0}'.format(e))
    if not nap_config:
        raise AutoprovisioningConfigError('autoprovisioning config file cannot be empty')
    nap_params = {'resourceLimits', 'serviceAccount', 'scopes', 'upgradeSettings', 'management', 'autoprovisioningLocations', 'minCpuPlatform', 'imageType', 'bootDiskKmsKey', 'diskSizeGb', 'diskType', 'shieldedInstanceConfig'}
    err = HasUnknownKeys(nap_config, nap_params)
    if err:
        raise AutoprovisioningConfigError(err)
    if nap_config.get('upgradeSettings'):
        upgrade_settings_params = {'maxSurgeUpgrade', 'maxUnavailableUpgrade'}
        err = HasUnknownKeys(nap_config.get('upgradeSettings'), upgrade_settings_params)
        if err:
            raise AutoprovisioningConfigError(err)
    if nap_config.get('management'):
        node_management_params = {'autoUpgrade', 'autoRepair'}
        err = HasUnknownKeys(nap_config.get('management'), node_management_params)
        if err:
            raise AutoprovisioningConfigError(err)
    if nap_config.get('shieldedInstanceConfig'):
        shielded_params = {'enableSecureBoot', 'enableIntegrityMonitoring'}
        err = HasUnknownKeys(nap_config.get('shieldedInstanceConfig'), shielded_params)
        if err:
            raise AutoprovisioningConfigError(err)