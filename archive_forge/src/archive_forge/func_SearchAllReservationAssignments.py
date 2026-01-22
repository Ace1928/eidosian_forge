from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Any, Dict, NamedTuple, Optional
from clients import utils as bq_client_utils
from utils import bq_error
def SearchAllReservationAssignments(client, location: str, job_type: str, assignee_type: str, assignee_id: str) -> Dict[str, Any]:
    """Searches reservations assignments for given assignee.

  Arguments:
    client: The client used to make the request.
    location: location of interest.
    job_type: type of job to be queried.
    assignee_type: Type of assignees for the reservation assignment.
    assignee_id: Project/folder/organization ID, to which the reservation is
      assigned.

  Returns:
    ReservationAssignment object if it exists.

  Raises:
    bq_error.BigqueryError: If required parameters are not passed in or
      reservation assignment not found.
  """
    if not location:
        raise bq_error.BigqueryError('location not specified.')
    if not job_type:
        raise bq_error.BigqueryError('job_type not specified.')
    if not assignee_type:
        raise bq_error.BigqueryError('assignee_type not specified.')
    if not assignee_id:
        raise bq_error.BigqueryError('assignee_id not specified.')
    assignee = '%ss/%s' % (assignee_type.lower(), assignee_id)
    query = 'assignee=%s' % assignee
    parent = 'projects/-/locations/%s' % location
    response = client.projects().locations().searchAllAssignments(parent=parent, query=query).execute()
    if 'assignments' in response:
        for assignment in response['assignments']:
            if assignment['jobType'] == job_type:
                return assignment
    raise bq_error.BigqueryError('Reservation assignment not found')