from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import boto
from gslib.cloud_api import ProjectIdException
def PopulateProjectId(project_id=None):
    """Returns the project_id from the boto config file if one is not provided."""
    if project_id:
        return project_id
    default_id = boto.config.get_value('GSUtil', 'default_project_id')
    if default_id:
        return default_id
    if UNIT_TEST_PROJECT_ID:
        return UNIT_TEST_PROJECT_ID
    raise ProjectIdException('MissingProjectId')