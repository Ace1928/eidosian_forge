from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc.batches import batch_submitter
from googlecloudsdk.command_lib.dataproc.batches import sparksql_batch_factory
Submit a Spark SQL batch job.