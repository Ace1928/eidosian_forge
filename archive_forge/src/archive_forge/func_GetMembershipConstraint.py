from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import api_util as fleet_api_util
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.core import exceptions
import six
def GetMembershipConstraint(client, messages, constraint_name, project_id, membership, release_track):
    """Returns a formatted membership constraint."""
    try:
        membership_obj = fleet_api_util.GetMembership(membership, release_track)
    except apitools_exceptions.HttpNotFoundError:
        raise exceptions.Error('Membership [{}] was not found in the Fleet.'.format(membership))
    try:
        request = messages.AnthospolicycontrollerstatusPaProjectsMembershipConstraintsGetRequest(name='projects/{}/membershipConstraints/{}/{}'.format(project_id, constraint_name, membership_obj.uniqueId))
        response = client.projects_membershipConstraints.Get(request)
    except apitools_exceptions.HttpNotFoundError:
        raise exceptions.Error('Constraint [{}] was not found in Fleet membership [{}].'.format(constraint_name, membership))
    return {'name': response.constraintRef.name, 'template': response.constraintRef.constraintTemplateName, 'enforcementAction': constants.get_enforcement_action_label(six.text_type(response.spec.enforcementAction)), 'membership': membership, 'violationCount': response.status.numViolations or 0, 'violations': [], 'match': response.spec.kubernetesMatch or {}, 'parameters': response.spec.parameters or {}}