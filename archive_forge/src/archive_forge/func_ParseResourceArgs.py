from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.fault_injection import jobs
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.fault_injection import flags
from googlecloudsdk.core import resources
@staticmethod
def ParseResourceArgs(args):
    """Parse, validate and return the CA and KMS key version args from the CLI.

    Args:
      args: The parsed arguments from the command-line.

    Returns:
      Tuple containing the Resource objects for the KMS key version and CA,
      respectively.
    """
    job_ref = args.CONCEPTS.job.Parse()
    exp_ref = resources.REGISTRY.Parse(args.experiment, collection='faultinjectiontesting.projects.locations.experiments', params={'projectsId': job_ref.projectsId, 'locationsId': job_ref.locationsId})
    if exp_ref.projectsId != job_ref.projectsId:
        raise exceptions.InvalidArgumentException('Experiment', 'Experiment must be in the same project as the JOBversion.')
    if exp_ref.locationsId != job_ref.locationsId:
        raise exceptions.InvalidArgumentException('Experiment', 'Experiment must be in the same location as the Jobversion.')
    return (job_ref, exp_ref)