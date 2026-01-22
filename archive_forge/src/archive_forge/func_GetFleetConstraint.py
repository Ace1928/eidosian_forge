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
def GetFleetConstraint(client, messages, constraint_name, project_id):
    """Returns a formatted Fleet constraint."""
    try:
        request = messages.AnthospolicycontrollerstatusPaProjectsFleetConstraintsGetRequest(name='projects/{}/fleetConstraints/{}'.format(project_id, constraint_name))
        response = client.projects_fleetConstraints.Get(request)
    except apitools_exceptions.HttpNotFoundError:
        raise exceptions.Error('Constraint [{}] was not found in the fleet.'.format(constraint_name))
    constraint = {'name': response.ref.name, 'template': response.ref.constraintTemplateName, 'violations': [], 'violationCount': response.numViolations or 0, 'memberships': [], 'membershipCount': response.numMemberships or 0}
    membership_constraints_request = messages.AnthospolicycontrollerstatusPaProjectsMembershipConstraintsListRequest(parent='projects/{}'.format(project_id))
    membership_constraints_response = client.projects_membershipConstraints.List(membership_constraints_request)
    for membership_constraint in membership_constraints_response.membershipConstraints:
        if constraint_name == '{}/{}'.format(membership_constraint.constraintRef.constraintTemplateName, membership_constraint.constraintRef.name):
            constraint['memberships'].append(membership_constraint.membershipRef.name)
    return constraint