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