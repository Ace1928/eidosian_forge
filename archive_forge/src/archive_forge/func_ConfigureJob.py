from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.command_lib.dataproc.jobs import spark
from googlecloudsdk.command_lib.dataproc.jobs import submitter
def ConfigureJob(self, messages, job, args):
    spark.SparkBase.ConfigureJob(messages, job, self.files_by_type, self.BuildLoggingConfig(messages, args.driver_log_levels), args)
    submitter.JobSubmitter.ConfigureJob(messages, job, args)