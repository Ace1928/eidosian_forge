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
def ParseTpuOptions(self, options, cluster):
    """Parses the options for TPUs."""
    if options.enable_tpu and (not options.enable_ip_alias):
        raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-ip-alias', opt='enable-tpu'))
    if not options.enable_tpu and options.tpu_ipv4_cidr:
        raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-tpu', opt='tpu-ipv4-cidr'))
    if not options.enable_tpu and options.enable_tpu_service_networking:
        raise util.Error(PREREQUISITE_OPTION_ERROR_MSG.format(prerequisite='enable-tpu', opt='enable-tpu-service-networking'))
    if options.enable_tpu:
        cluster.enableTpu = options.enable_tpu
        if options.enable_tpu_service_networking:
            tpu_config = self.messages.TpuConfig(enabled=options.enable_tpu, ipv4CidrBlock=options.tpu_ipv4_cidr, useServiceNetworking=options.enable_tpu_service_networking)
            cluster.tpuConfig = tpu_config