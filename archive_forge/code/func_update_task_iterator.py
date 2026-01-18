from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import errors_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import stdin_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks import task_graph_executor
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.buckets import update_bucket_task
def update_task_iterator(self, args):
    user_request_args = user_request_args_factory.get_user_request_args_from_command_args(args, metadata_type=user_request_args_factory.MetadataType.BUCKET)
    if user_request_args_factory.adds_or_removes_acls(user_request_args):
        fields_scope = cloud_api.FieldsScope.FULL
    else:
        fields_scope = cloud_api.FieldsScope.NO_ACL
    urls = stdin_iterator.get_urls_iterable(args.url, args.read_paths_from_stdin)
    for url_string in urls:
        url = storage_url.storage_url_from_string(url_string)
        errors_util.raise_error_if_not_bucket(args.command_path, url)
        for resource in wildcard_iterator.get_wildcard_iterator(url_string, fields_scope=fields_scope, get_bucket_metadata=_is_initial_bucket_metadata_needed(user_request_args)):
            yield update_bucket_task.UpdateBucketTask(resource, user_request_args=user_request_args)