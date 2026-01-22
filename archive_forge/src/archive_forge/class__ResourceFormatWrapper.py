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
class _ResourceFormatWrapper(list_util.BaseFormatWrapper):
    """For formatting how resources print when listed."""

    def __init__(self, resource, display_detail=list_util.DisplayDetail.SHORT, full_formatter=None, include_etag=False, object_state=None, readable_sizes=False, use_gsutil_style=False):
        """See list_util.BaseFormatWrapper class for function doc strings."""
        super(_ResourceFormatWrapper, self).__init__(resource, display_detail=display_detail, include_etag=include_etag, object_state=object_state, readable_sizes=readable_sizes, use_gsutil_style=use_gsutil_style)
        self._full_formatter = full_formatter

    def _format_for_list_long(self):
        """Returns string of select properties from resource."""
        if isinstance(self.resource, resource_reference.PrefixResource):
            return LONG_LIST_ROW_FORMAT.format(size='', creation_time='', url=self.resource.storage_url.url_string, metageneration='', etag='')
        creation_time = resource_util.get_formatted_timestamp_in_utc(self.resource.creation_time)
        url_string, metageneration_string = self._check_and_handles_versions()
        if self._include_etag:
            etag_string = '  etag={}'.format(str(self.resource.etag))
        else:
            etag_string = ''
        return LONG_LIST_ROW_FORMAT.format(size=list_util.check_and_convert_to_readable_sizes(self.resource.size, self._readable_sizes, self._use_gsutil_style), creation_time=creation_time, url=url_string, metageneration=metageneration_string, etag=etag_string)

    def __str__(self):
        if self._display_detail == list_util.DisplayDetail.LONG and (isinstance(self.resource, resource_reference.ObjectResource) or isinstance(self.resource, resource_reference.PrefixResource)):
            return self._format_for_list_long()
        show_version_in_url = self._object_state in (cloud_api.ObjectState.LIVE_AND_NONCURRENT, cloud_api.ObjectState.SOFT_DELETED)
        if self._display_detail == list_util.DisplayDetail.FULL and (isinstance(self.resource, resource_reference.BucketResource) or isinstance(self.resource, resource_reference.ObjectResource)):
            return self._full_formatter.format(self.resource, show_version_in_url=show_version_in_url)
        if self._display_detail == list_util.DisplayDetail.JSON:
            return self.resource.get_json_dump()
        if show_version_in_url:
            return self.resource.storage_url.url_string
        return self.resource.storage_url.versionless_url_string