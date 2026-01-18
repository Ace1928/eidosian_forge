import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'), ('duration',))
@exc.on_http_error(exc.GitlabTimeTrackingError)
def time_estimate(self, duration: str, **kwargs: Any) -> Dict[str, Any]:
    """Set an estimated time of work for the object.

        Args:
            duration: Duration in human format (e.g. 3h30)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
    path = f'{self.manager.path}/{self.encoded_id}/time_estimate'
    data = {'duration': duration}
    result = self.manager.gitlab.http_post(path, post_data=data, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result