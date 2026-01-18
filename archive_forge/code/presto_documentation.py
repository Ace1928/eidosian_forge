from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import presto
from googlecloudsdk.command_lib.dataproc.jobs import submitter
Submit a Presto job to a cluster.

  Submit a Presto job to a cluster

  ## EXAMPLES

  To submit a Presto job with a local script, run:

    $ {command} --cluster=my-cluster --file=my_script.R

  To submit a Presto job with inline queries, run:

    $ {command} --cluster=my-cluster -e="SELECT * FROM foo WHERE bar > 2"
  