from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import folder_util
from googlecloudsdk.command_lib.storage import plurality_checkable_iterator
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import shim_format_util
import six
def _check_and_handles_versions(self):
    show_version_in_url = self._object_state in (cloud_api.ObjectState.LIVE_AND_NONCURRENT, cloud_api.ObjectState.SOFT_DELETED)
    if show_version_in_url:
        url_string = self.resource.storage_url.url_string
        metageneration_string = '  metageneration={}'.format(six.text_type(self.resource.metageneration))
    else:
        url_string = self.resource.storage_url.versionless_url_string
        metageneration_string = ''
    return (url_string, metageneration_string)