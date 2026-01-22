from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ExtractJobRef(args):
    """Extract the Job Ref for a command. Used with ArgsForJobRef.

  Args:
    args: The command line arguments.
  Returns:
    A Job resource.
  """
    job = args.job
    region = dataflow_util.GetRegion(args)
    return resources.REGISTRY.Parse(job, params={'projectId': properties.VALUES.core.project.GetOrFail, 'location': region}, collection='dataflow.projects.locations.jobs')