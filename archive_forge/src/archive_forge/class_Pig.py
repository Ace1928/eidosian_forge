from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import pig
from googlecloudsdk.command_lib.dataproc.jobs import submitter
class Pig(pig.PigBase, submitter.JobSubmitter):
    """Submit a Pig job to a cluster.

  Submit a Pig job to a cluster.

  ## EXAMPLES

  To submit a Pig job with a local script, run:

    $ {command} --cluster=my-cluster --file=my_queries.pig

  To submit a Pig job with inline queries, run:

    $ {command} --cluster=my-cluster
        -e="LNS = LOAD 'gs://my_bucket/my_file.txt' AS (line)"
        -e="WORDS = FOREACH LNS GENERATE FLATTEN(TOKENIZE(line)) AS word"
        -e="GROUPS = GROUP WORDS BY word"
        -e="WORD_COUNTS = FOREACH GROUPS GENERATE group, COUNT(WORDS)"
        -e="DUMP WORD_COUNTS"
  """

    @staticmethod
    def Args(parser):
        pig.PigBase.Args(parser)
        submitter.JobSubmitter.Args(parser)

    def ConfigureJob(self, messages, job, args):
        pig.PigBase.ConfigureJob(messages, job, self.files_by_type, self.BuildLoggingConfig(messages, args.driver_log_levels), args)
        submitter.JobSubmitter.ConfigureJob(messages, job, args)