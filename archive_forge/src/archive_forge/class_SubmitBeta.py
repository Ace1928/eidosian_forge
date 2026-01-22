from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.protorpclite.messages import DecodeError
from apitools.base.py import encoding
from googlecloudsdk.api_lib.batch import jobs
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.batch import resource_args
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class SubmitBeta(Submit):
    """Submit a Batch job.

  This command creates and submits a Batch job. After you create and
  submit the job, Batch automatically queues, schedules, and executes it.

  ## EXAMPLES

  To submit a job with a sample JSON configuration file (config.json) and name
  `projects/foo/locations/us-central1/jobs/bar`, run:

    $ {command} projects/foo/locations/us-central1/jobs/bar --config=config.json

  To submit a job with a sample YAML configuration file (config.yaml) and
  name projects/foo/locations/us-central1/jobs/bar, run:

    $ {command} projects/foo/locations/us-central1/jobs/bar --config=config.yaml

  To submit a job through stdin with a sample job configuration and name
  `projects/foo/locations/us-central1/jobs/bar`, run:

    $ {command} projects/foo/locations/us-central1/jobs/bar --config=-

      then input json job config via stdin
      {
        job config
      }

  To submit a job through HereDoc with a sample job configuration and name
  `projects/foo/locations/us-central1/jobs/bar`, run:

    $ {command} projects/foo/locations/us-central1/jobs/bar --config=- << EOF

      {
        job config
      }
      EOF

  For details about how to define a job's configuration using JSON, see the
  projects.locations.jobs resource in the Batch API Reference.
  If you want to define a job's configuration using YAML, convert the JSON
  syntax to YAML.
  """