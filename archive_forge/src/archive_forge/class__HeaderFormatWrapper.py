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
class _HeaderFormatWrapper(list_util.BaseFormatWrapper):
    """For formatting how headers print when listed."""

    def __init__(self, resource, display_detail=list_util.DisplayDetail.SHORT, include_etag=False, object_state=None, readable_sizes=False, full_formatter=None, use_gsutil_style=False):
        """See list_util.BaseFormatWrapper class for function doc strings."""
        super(_HeaderFormatWrapper, self).__init__(resource, display_detail=display_detail, full_formatter=full_formatter, include_etag=include_etag, object_state=object_state, readable_sizes=readable_sizes, use_gsutil_style=use_gsutil_style)

    def __str__(self):
        if self._use_gsutil_style and isinstance(self.resource, resource_reference.BucketResource):
            return ''
        url = self.resource.storage_url.versionless_url_string
        if self._display_detail == list_util.DisplayDetail.JSON:
            return self.resource.get_json_dump()
        return '\n{}:'.format(url)