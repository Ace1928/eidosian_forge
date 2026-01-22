from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import functools
import ipaddress
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import containers_utils
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import disks_util
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.zones import service as zones_service
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as core_resources
import six
def AddLocalNvdimmArgs(parser):
    """Adds local NVDIMM argument for instances and instance-templates."""
    parser.add_argument('--local-nvdimm', type=arg_parsers.ArgDict(spec={'size': arg_parsers.BinarySize()}), action='append', help="      Attaches a local NVDIMM to the instances.\n\n      *size*::: Optional. Size of the NVDIMM disk. The value must be a whole\n      number followed by a size unit of ``KB'' for kilobyte, ``MB'' for\n      megabyte, ``GB'' for gigabyte, or ``TB'' for terabyte. For example,\n      ``3TB'' will produce a 3 terabyte disk. Allowed values are: 3TB and 6TB\n      and the default is 3 TB.\n      ")