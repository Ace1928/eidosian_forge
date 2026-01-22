from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _get_user_assignment_types(self):
    return [AssignmentType.USER_PROJECT, AssignmentType.USER_DOMAIN]