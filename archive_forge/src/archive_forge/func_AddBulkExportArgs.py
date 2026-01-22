from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddBulkExportArgs(parser):
    """Adds flags for the bulk-export command."""
    AddOnErrorFlag(parser)
    AddPathFlag(parser)
    AddFormatFlag(parser)
    resource_storage_mutex = parser.add_group(mutex=True, help='Select `storage-path` if you want to specify the Google Cloud Storage bucket bulk-export should use for Cloud Asset Inventory Export. Alternatively, you can provide a `RESOURCE TYPE FILTER` to filter resources. Filtering resources _does not_ use Google Cloud Storage to export resources.')
    AddResourceTypeFlags(resource_storage_mutex)
    resource_storage_mutex.add_argument('--storage-path', required=False, help='Google Cloud Storage path where a Cloud Asset Inventory export will be stored, example: `gs://your-bucket-name/your/prefix/path`')
    _GetBulkExportParentGroup(parser)