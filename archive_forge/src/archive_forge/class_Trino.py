from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import submitter
from googlecloudsdk.command_lib.dataproc.jobs import trino
class Trino(trino.TrinoBase, submitter.JobSubmitter):
    """Submit a Trino job to a cluster.

  Submit a Trino job to a cluster

  ## EXAMPLES

  To submit a Trino job with a local script, run:

    $ {command} --cluster=my-cluster --file=my_script.R

  To submit a Trino job with inline queries, run:

    $ {command} --cluster=my-cluster -e="SELECT * FROM foo WHERE bar > 2"
  """

    @staticmethod
    def Args(parser):
        trino.TrinoBase.Args(parser)
        submitter.JobSubmitter.Args(parser)

    def ConfigureJob(self, messages, job, args):
        trino.TrinoBase.ConfigureJob(messages, job, self.files_by_type, self.BuildLoggingConfig(messages, args.driver_log_levels), args)
        submitter.JobSubmitter.ConfigureJob(messages, job, args)