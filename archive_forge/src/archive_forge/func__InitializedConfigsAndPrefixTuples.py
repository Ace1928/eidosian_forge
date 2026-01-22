from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _InitializedConfigsAndPrefixTuples(self):
    """Returns the initialized configs as a list of (config, prefix) tuples."""
    all_configs_and_prefixes = [(self.retry_config, self.retry_config_mask_prefix), (self.rate_limits, self.rate_limits_mask_prefix), (self.app_engine_routing_override, self.app_engine_routing_override_mask_prefix), (self.http_target, self.http_target_mask_prefix), (self.stackdriver_logging_config, self.stackdriver_logging_config_mask_prefix)]
    return [(config, prefix) for config, prefix in all_configs_and_prefixes if config]