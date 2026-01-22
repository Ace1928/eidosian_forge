from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetProjectIdArgument(verb):
    """Return the PROJECT_ID argument for XPN commands."""
    arg = base.Argument('project', metavar='PROJECT_ID', help='ID for the project to {verb}'.format(verb=verb))
    return arg