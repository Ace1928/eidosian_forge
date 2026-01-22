from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import utils as api_utils
from googlecloudsdk.api_lib.dataproc import compute_helpers
from googlecloudsdk.api_lib.dataproc import constants
from googlecloudsdk.api_lib.dataproc import exceptions
from googlecloudsdk.api_lib.dataproc import storage_helpers
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import times
import six
from stdin.
def AddSecureMultiTenancyGroup(parser):
    """Adds the argument group to handle Secure Multi-Tenancy configurations."""
    secure_multi_tenancy_group = parser.add_argument_group(mutex=True, help='Specifying these flags will enable Secure Multi-Tenancy for the cluster.')
    secure_multi_tenancy_group.add_argument('--identity-config-file', help='Path to a YAML (or JSON) file containing the configuration for Secure Multi-Tenancy\non the cluster. The path can be a Cloud Storage URL (Example: \'gs://path/to/file\')\nor a local file system path. If you pass "-" as the value of the flag the file content\nwill be read from stdin.\n\nThe YAML file is formatted as follows:\n\n```\n  # Required. The mapping from user accounts to service accounts.\n  user_service_account_mapping:\n    bob@company.com: service-account-bob@project.iam.gserviceaccount.com\n    alice@company.com: service-account-alice@project.iam.gserviceaccount.com\n```\n        ')
    secure_multi_tenancy_group.add_argument('--secure-multi-tenancy-user-mapping', help=textwrap.dedent('          A string of user-to-service-account mappings. Mappings are separated\n          by commas, and each mapping takes the form of\n          "user-account:service-account". Example:\n          "bob@company.com:service-account-bob@project.iam.gserviceaccount.com,alice@company.com:service-account-alice@project.iam.gserviceaccount.com".'))