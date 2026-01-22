from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetPackagePathFlag(metavar=None):
    return GetFlagOrPositional(name='package_path', positional=False, required=False, help="      Path to remote subdirectory containing Kubernetes Resource configuration\n      files or directories.\n      Defaults to the root directory.\n      Uses '/' as the path separator (regardless of OS).\n      e.g. staging/cockroachdb\n      ", metavar=metavar)