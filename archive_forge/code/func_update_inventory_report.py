from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.api_lib.storage.gcs_json import client as gcs_json_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
def update_inventory_report(self, report_config_name, destination_url=None, metadata_fields=None, start_date=None, end_date=None, frequency=None, csv_separator=None, csv_delimiter=None, csv_header=None, parquet=None, display_name=None):
    """Updates a report config.

    Args:
      report_config_name (str): The name of the report config to be modified.
      destination_url (storage_url.CloudUrl): The destination url where the
        generated reports will be stored.
      metadata_fields (list[str]): Fields to be included in the report.
      start_date (datetime.datetime.date): The date to start generating reports.
      end_date (datetime.datetime.date): The date after which to stop generating
        reports.
      frequency (str): Can be either DAILY or WEEKLY.
      csv_separator (str): The character used to separate the records in the
        CSV file.
      csv_delimiter (str): The delimiter that separates the fields in the CSV
        file.
      csv_header (bool): If True, include the headers in the CSV file.
      parquet (bool): If True, set the parquet options.
      display_name (str): Display name for the report config.

    Returns:
      The created ReportConfig object.
    """
    frequency_options, frequency_update_mask = self._get_frequency_options_and_update_mask(start_date, end_date, frequency)
    object_metadata_report_options, metadata_update_mask = self._get_metadata_options_and_update_mask(metadata_fields, destination_url)
    report_format_options, report_format_mask = self._get_report_format_options_and_update_mask(csv_separator, csv_delimiter, csv_header, parquet)
    update_mask = frequency_update_mask + metadata_update_mask + report_format_mask
    if display_name is not None:
        update_mask.append('displayName')
    if not update_mask:
        raise errors.CloudApiError('Nothing to update for report config: {}'.format(report_config_name))
    report_config = self.messages.ReportConfig(csvOptions=report_format_options.csv, parquetOptions=report_format_options.parquet, displayName=display_name, frequencyOptions=frequency_options, objectMetadataReportOptions=object_metadata_report_options)
    request = self.messages.StorageinsightsProjectsLocationsReportConfigsPatchRequest(name=report_config_name, reportConfig=report_config, updateMask=','.join(update_mask))
    return self.client.projects_locations_reportConfigs.Patch(request)