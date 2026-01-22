from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def ValidateIstioConfigUpdateArgs(istio_config_args, disable_addons_args):
    """Validates flags specifying Istio config for update.

  Args:
    istio_config_args: parsed comandline arguments for --istio_config.
    disable_addons_args: parsed comandline arguments for --update-addons.

  Raises:
    InvalidArgumentException: --update-addons=Istio=ENABLED
    or --istio_config is specified
  """
    if disable_addons_args and disable_addons_args.get('Istio'):
        return
    if istio_config_args or (disable_addons_args and disable_addons_args.get('Istio') is False):
        raise exceptions.InvalidArgumentException('--istio-config', 'The Istio addon is no longer supported.')