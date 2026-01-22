from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import validation
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.core.util import files
def ValidateLocalRunArgs(args):
    """Validates the arguments specified in `local-run` command and normalize them."""
    args_local_package_pach = args.local_package_path
    if args_local_package_pach:
        work_dir = os.path.abspath(files.ExpandHomeDir(args_local_package_pach))
        if not os.path.exists(work_dir) or not os.path.isdir(work_dir):
            raise exceptions.InvalidArgumentException('--local-package-path', "Directory '{}' is not found.".format(work_dir))
    else:
        work_dir = files.GetCWD()
    args.local_package_path = work_dir
    _ValidBuildArgsOfLocalRun(args)
    return args