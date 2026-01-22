from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import properties
def GetParentNameForGetHistory(organization, project, attribute='root cloud asset'):
    """Gets the parent name from organization Id, project Id."""
    VerifyParentForGetHistory(organization, project, attribute)
    if organization:
        return 'organizations/{0}'.format(organization)
    return 'projects/{0}'.format(project)