from keystone.assignment.backends import base
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
def _get_group_assignment_types(self):
    return [AssignmentType.GROUP_PROJECT, AssignmentType.GROUP_DOMAIN]