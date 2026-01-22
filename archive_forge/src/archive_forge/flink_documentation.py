from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.command_lib.dataproc.jobs import flink
from googlecloudsdk.command_lib.dataproc.jobs import submitter
Submit a Flink job to a cluster.

  Submit a Flink job to a cluster.

  ## EXAMPLES

  To submit a Flink job that runs the main class of a jar, run:

    $ {command} --cluster=my-cluster --region=us-central1 --jar=my_jar.jar -- arg1 arg2

  To submit a Flink job that runs a specific class as an entrypoint:

    $ {command} --cluster=my-cluster --region=us-central1  --class=org.my.main.Class  \
      --jars=my_jar.jar -- arg1 arg2

  To submit a Flink job that runs a jar that is on the cluster, run:

    $ {command} --cluster=my-cluster --region=us-central1 \
        --jar=/usr/lib/flink/examples/streaming/TopSpeedWindowing.jar

  