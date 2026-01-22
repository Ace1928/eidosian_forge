from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import rm_command_util
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_task_iterator
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def add_cp_mv_rsync_flags(parser):
    """Adds flags shared by cp, mv, and rsync."""
    flags.add_additional_headers_flag(parser)
    flags.add_continue_on_error_flag(parser)
    flags.add_object_metadata_flags(parser)
    flags.add_precondition_flags(parser)
    parser.add_argument('--content-md5', metavar='MD5_DIGEST', help='Manually specified MD5 hash digest for the contents of an uploaded file. This flag cannot be used when uploading multiple files. The custom digest is used by the cloud provider for validation.')
    parser.add_argument('-n', '--no-clobber', action='store_true', help='Do not overwrite existing files or objects at the destination. Skipped items will be printed. This option may perform an additional GET request for cloud objects before attempting an upload.')
    parser.add_argument('-P', '--preserve-posix', action='store_true', help=_PRESERVE_POSIX_HELP_TEXT)
    parser.add_argument('-U', '--skip-unsupported', action='store_true', help='Skip objects with unsupported object types.')