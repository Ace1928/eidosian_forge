import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
def _get_update_method(self) -> Callable[..., Union[Dict[str, Any], requests.Response]]:
    """Return the HTTP method to use.

        Returns:
            http_put (default) or http_post
        """
    if self._update_method is UpdateMethod.POST:
        http_method = self.manager.gitlab.http_post
    else:
        http_method = self.manager.gitlab.http_put
    return http_method