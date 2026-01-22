from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.command_lib.dns import util as command_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
def _MakeDnssecConfig(args, messages, api_version='v1'):
    """Parse user-specified args into a DnssecConfig message."""
    dnssec_config = None
    if args.dnssec_state is not None:
        dnssec_config = command_util.ParseDnssecConfigArgs(args, messages, api_version)
    else:
        bad_args = ['denial_of_existence', 'ksk_algorithm', 'zsk_algorithm', 'ksk_key_length', 'zsk_key_length']
        for bad_arg in bad_args:
            if getattr(args, bad_arg, None) is not None:
                raise exceptions.InvalidArgumentException(bad_arg, 'DNSSEC must be enabled in order to use other DNSSEC arguments. Please set --dnssec-state to "on" or "transfer".')
    return dnssec_config