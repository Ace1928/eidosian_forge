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
def add_cp_and_mv_flags(parser):
    """Adds flags to cp, mv, or other cp-based commands."""
    parser.add_argument('source', nargs='*', help='The source path(s) to copy.')
    parser.add_argument('destination', help='The destination path.')
    add_cp_mv_rsync_flags(parser)
    parser.add_argument('-A', '--all-versions', action='store_true', help=_ALL_VERSIONS_HELP_TEXT)
    parser.add_argument('--do-not-decompress', action='store_true', help='Do not automatically decompress downloaded gzip files.')
    parser.add_argument('-D', '--daisy-chain', action='store_true', help='Copy in "daisy chain" mode, which means copying an object by first downloading it to the machine where the command is run, then uploading it to the destination bucket. The default mode is a "copy in the cloud," where data is copied without uploading or downloading. During a copy in the cloud, a source composite object remains composite at its destination. However, you can use daisy chain mode to change a composite object into a non-composite object. Note: Daisy chain mode is automatically used when copying between providers.')
    add_include_managed_folders_flag(parser)
    symlinks_group = parser.add_group(mutex=True, help='Flags to influence behavior when handling symlinks. Only one value may be set.')
    add_ignore_symlinks_flag(symlinks_group)
    add_preserve_symlinks_flag(symlinks_group)
    parser.add_argument('-L', '--manifest-path', help=_MANIFEST_HELP_TEXT)
    parser.add_argument('-v', '--print-created-message', action='store_true', help='Prints the version-specific URL for each copied object.')
    parser.add_argument('-s', '--storage-class', help='Specify the storage class of the destination object. If not specified, the default storage class of the destination bucket is used. This option is not valid for copying to non-cloud destinations.')
    gzip_flags_group = parser.add_group(mutex=True)
    add_gzip_in_flight_flags(gzip_flags_group)
    gzip_flags_group.add_argument('-Z', '--gzip-local-all', action='store_true', help=_GZIP_LOCAL_ALL_HELP_TEXT)
    gzip_flags_group.add_argument('-z', '--gzip-local', metavar='FILE_EXTENSIONS', type=arg_parsers.ArgList(), help=_GZIP_LOCAL_EXTENSIONS_HELP_TEXT)
    acl_flags_group = parser.add_group()
    flags.add_predefined_acl_flag(acl_flags_group)
    flags.add_preserve_acl_flag(acl_flags_group)
    flags.add_encryption_flags(parser)
    flags.add_read_paths_from_stdin_flag(parser, help_text='Read the list of resources to copy from stdin. No need to enter a source argument if this flag is present.\nExample: "storage cp -I gs://bucket/destination"\n Note: To copy the contents of one file directly from stdin, use "-" as the source argument without the "-I" flag.')