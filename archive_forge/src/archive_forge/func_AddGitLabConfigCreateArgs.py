from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddGitLabConfigCreateArgs(parser):
    """Set up all the argparse flags for creating a GitLab config.

  Args:
    parser: An argparse.ArgumentParser-like object.

  Returns:
    The parser argument with GitLab config flags added in.
  """
    return AddGitLabConfigArgs(parser, update=False)