from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddWorkerpoolArgs(parser, release_track, update=False):
    """Set up all the argparse flags for creating or updating a workerpool.

  Args:
    parser: An argparse.ArgumentParser-like object.
    release_track: A base.ReleaseTrack-like object.
    update: If true, use the version of the flags for updating a workerpool.
      Otherwise, use the version for creating a workerpool.

  Returns:
    The parser argument with workerpool flags added in.
  """
    verb = 'update' if update else 'create'
    parser.add_argument('WORKER_POOL', help='Unique identifier for the worker pool to %s. This value should be 1-63 characters, and valid characters are [a-z][0-9]-' % verb)
    parser.add_argument('--region', required=True, help='Cloud region where the worker pool is %sd. See https://cloud.google.com/build/docs/locations for available locations.' % verb)
    file_or_flags = parser.add_mutually_exclusive_group(required=update)
    if release_track != base.ReleaseTrack.ALPHA:
        file_or_flags.add_argument('--config-from-file', help=_UPDATE_FILE_DESC if update else _CREATE_FILE_DESC)
    else:
        file_or_flags.add_argument('--config-from-file', help=_UPDATE_FILE_DESC_ALPHA if update else _CREATE_FILE_DESC_ALPHA)
    flags = file_or_flags.add_argument_group('Command-line flags to configure the private pool:')
    if not update:
        flags.add_argument('--peered-network', help='Existing network to which workers are peered. The network is specified in\nresource URL format\nprojects/{network_project}/global/networks/{network_name}.\n\nIf not specified, the workers are not peered to any network.\n')
    if not update:
        flags.add_argument('--peered-network-ip-range', help='An IP range for your peered network. Specify the IP range using Classless\nInter-Domain Routing (CIDR) notation with a slash and the subnet prefix size,\nsuch as `/29`.\n\nYour subnet prefix size must be between 1 and 29.  Optional: you can specify an\nIP address before the subnet prefix value - for example `192.168.0.0/24`.\n\nIf no IP address is specified, your VPC automatically determines the starting\nIP for the range. If no IP range is specified, Cloud Build uses `/24` as the\ndefault network IP range.\n')
    worker_flags = flags.add_argument_group('Configuration to be used for creating workers in the worker pool:')
    worker_flags.add_argument('--worker-machine-type', help='Compute Engine machine type for a worker pool.\n\nIf unspecified, Cloud Build uses a standard machine type.\n')
    worker_flags.add_argument('--worker-disk-size', type=arg_parsers.BinarySize(lower_bound='100GB'), help='Size of the disk attached to the worker.\n\nIf not given, Cloud Build will use a standard disk size.\n')
    if release_track == base.ReleaseTrack.GA:
        worker_flags.add_argument('--no-external-ip', hidden=release_track == base.ReleaseTrack.GA, action=actions.DeprecationAction('--no-external-ip', warn='The `--no-external-ip` option is deprecated; use `--no-public-egress` and/or `--public-egress instead`.', removed=False, action='store_true'), help='  If set, workers in the worker pool are created without an external IP address.\n\n  If the worker pool is within a VPC Service Control perimeter, use this flag.\n  ')
    if update:
        egress_flags = flags.add_mutually_exclusive_group()
        egress_flags.add_argument('--no-public-egress', action='store_true', help='If set, workers in the worker pool are created without an external IP address.\n\nIf the worker pool is within a VPC Service Control perimeter, use this flag.\n  ')
        egress_flags.add_argument('--public-egress', action='store_true', help='If set, workers in the worker pool are created with an external IP address.\n')
    else:
        flags.add_argument('--no-public-egress', action='store_true', help='If set, workers in the worker pool are created without an external IP address.\n\nIf the worker pool is within a VPC Service Control perimeter, use this flag.\n')
    return parser