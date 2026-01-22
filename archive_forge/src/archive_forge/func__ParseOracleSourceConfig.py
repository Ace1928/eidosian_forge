from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datastream import exceptions as ds_exceptions
from googlecloudsdk.api_lib.datastream import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
def _ParseOracleSourceConfig(self, oracle_source_config_file, release_track):
    """Parses a oracle_sorce_config into the OracleSourceConfig message."""
    if release_track == base.ReleaseTrack.BETA:
        return self._ParseOracleSourceConfigBeta(oracle_source_config_file, release_track)
    return util.ParseMessageAndValidateSchema(oracle_source_config_file, 'OracleSourceConfig', self._messages.OracleSourceConfig)