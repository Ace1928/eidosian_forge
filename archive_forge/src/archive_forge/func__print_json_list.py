from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import list_util
from googlecloudsdk.command_lib.storage.resources import gcloud_full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import gsutil_full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.command_lib.storage.resources import shim_format_util
def _print_json_list(self, resource_wrappers):
    """Prints ResourceWrapper objects as JSON list."""
    is_empty_list = True
    for i, resource_wrapper in enumerate(resource_wrappers):
        is_empty_list = False
        if i == 0:
            print('[')
            print(resource_wrapper, end='')
        else:
            print(',\n{}'.format(resource_wrapper), end='')
    print()
    if not is_empty_list:
        print(']')