from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc.jobs import submitter
from googlecloudsdk.command_lib.dataproc.jobs import trino
Submit a Trino job to a cluster.

  Submit a Trino job to a cluster

  ## EXAMPLES

  To submit a Trino job with a local script, run:

    $ {command} --cluster=my-cluster --file=my_script.R

  To submit a Trino job with inline queries, run:

    $ {command} --cluster=my-cluster -e="SELECT * FROM foo WHERE bar > 2"
  