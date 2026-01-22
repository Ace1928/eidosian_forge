from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from absl import flags
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
def RunWithArgs(self, identifier: str, destination_uris: str) -> Optional[int]:
    """Perform an extract operation of source into destination_uris.

    Usage:
      extract <source_table> <destination_uris>

    Use -m option to extract a source_model.

    Examples:
      bq extract ds.table gs://mybucket/table.csv
      bq extract -m ds.model gs://mybucket/model

    Arguments:
      source_table: Source table to extract.
      source_model: Source model to extract.
      destination_uris: One or more Google Cloud Storage URIs, separated by
        commas.
    """
    client = bq_cached_client.Client.Get()
    kwds = {'job_id': frontend_utils.GetJobIdFromFlags()}
    if FLAGS.location:
        kwds['location'] = FLAGS.location
    if self.m:
        reference = client.GetModelReference(identifier)
    else:
        reference = client.GetTableReference(identifier)
    job = client.Extract(reference, destination_uris, print_header=self.print_header, field_delimiter=frontend_utils.NormalizeFieldDelimiter(self.field_delimiter), destination_format=self.destination_format, trial_id=self.trial_id, add_serving_default_signature=self.add_serving_default_signature, compression=self.compression, use_avro_logical_types=self.use_avro_logical_types, **kwds)
    if FLAGS.sync:
        frontend_utils.PrintJobMessages(bq_client_utils.FormatJobInfo(job))
    else:
        self.PrintJobStartInfo(job)