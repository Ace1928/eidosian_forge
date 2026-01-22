from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import hive
from googlecloudsdk.command_lib.dataproc.jobs import submitter
class Hive(hive.HiveBase, submitter.JobSubmitter):
    """Submit a Hive job to a cluster.

  Submit a Hive job to a cluster.

  ## EXAMPLES

  To submit a Hive job with a local script, run:

    $ {command} --cluster=my-cluster --file=my_queries.q

  To submit a Hive job with inline queries, run:

    $ {command} --cluster=my-cluster
        -e="CREATE EXTERNAL TABLE foo(bar int) LOCATION 'gs://my_bucket/'"
        -e="SELECT * FROM foo WHERE bar > 2"
  """

    @classmethod
    def Args(cls, parser):
        hive.HiveBase.Args(parser)
        submitter.JobSubmitter.Args(parser)

    def ConfigureJob(self, messages, job, args):
        hive.HiveBase.ConfigureJob(messages, job, self.files_by_type, args)
        submitter.JobSubmitter.ConfigureJob(messages, job, args)