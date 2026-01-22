from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import notifications
def _delete_application_credentials_for_user_on_project(self, user_id, project_id):
    """Delete all application credentials for a user on a given project.

        :param str user_id: User ID
        :param str project_id: Project ID

        This is triggered when a user loses a role assignment on a project.
        """
    hints = driver_hints.Hints()
    hints.add_filter('project_id', project_id)
    app_creds = self.driver.list_application_credentials_for_user(user_id, hints)
    self.driver.delete_application_credentials_for_user_on_project(user_id, project_id)
    for app_cred in app_creds:
        self.get_application_credential.invalidate(self, app_cred['id'])