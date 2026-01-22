from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
def _AppendAwsArgs(args, import_args):
    """Appends args related to AWS image import."""
    daisy_utils.AppendArg(import_args, 'aws_access_key_id', args.aws_access_key_id)
    daisy_utils.AppendArg(import_args, 'aws_secret_access_key', args.aws_secret_access_key)
    daisy_utils.AppendArg(import_args, 'aws_session_token', args.aws_session_token)
    daisy_utils.AppendArg(import_args, 'aws_region', args.aws_region)
    if args.aws_ami_id:
        daisy_utils.AppendArg(import_args, 'aws_ami_id', args.aws_ami_id)
    if args.aws_ami_export_location:
        daisy_utils.AppendArg(import_args, 'aws_ami_export_location', args.aws_ami_export_location)
    if args.aws_source_ami_file_path:
        daisy_utils.AppendArg(import_args, 'aws_source_ami_file_path', args.aws_source_ami_file_path)