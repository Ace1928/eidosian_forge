from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _ValidateDumpPath(self, args):
    if args.dump_path is None:
        return
    try:
        storage_util.ObjectReference.FromArgument(args.dump_path, allow_empty_object=False)
    except Exception as e:
        raise exceptions.InvalidArgumentException('dump-path', six.text_type(e))