from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc.batches import (
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class Submit(base.Group):
    """Submit a Dataproc batch job."""
    detailed_help = {'EXAMPLES': "          To submit a PySpark job, run:\n\n            $ {command} pyspark my-pyspark.py --region='us-central1' --deps-bucket=gs://my-bucket --py-files='path/to/my/python/scripts'\n\n          To submit a Spark job, run:\n\n            $ {command} spark --region='us-central1' --deps-bucket=gs://my-bucket --jar='my_jar.jar' -- ARG1 ARG2\n\n          To submit a Spark-R job, run:\n\n            $ {command} spark-r my-main-r.r --region='us-central1' --deps-bucket=gs://my-bucket -- ARG1 ARG2\n\n          To submit a Spark-Sql job, run:\n\n            $ {command} spark-sql 'my-sql-script.sql' --region='us-central1' --deps-bucket=gs://my-bucket --vars='variable=value'\n        "}

    @staticmethod
    def Args(parser):
        flags.AddAsync(parser)
        batches_create_request_factory.AddArguments(parser, dp.Dataproc().api_version)