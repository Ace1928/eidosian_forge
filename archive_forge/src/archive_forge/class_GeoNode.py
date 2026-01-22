from typing import Any, cast, Dict, List, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GeoNode(SaveMixin, ObjectDeleteMixin, RESTObject):

    @cli.register_custom_action('GeoNode')
    @exc.on_http_error(exc.GitlabRepairError)
    def repair(self, **kwargs: Any) -> None:
        """Repair the OAuth authentication of the geo node.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabRepairError: If the server failed to perform the request
        """
        path = f'/geo_nodes/{self.encoded_id}/repair'
        server_data = self.manager.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        self._update_attrs(server_data)

    @cli.register_custom_action('GeoNode')
    @exc.on_http_error(exc.GitlabGetError)
    def status(self, **kwargs: Any) -> Dict[str, Any]:
        """Get the status of the geo node.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The status of the geo node
        """
        path = f'/geo_nodes/{self.encoded_id}/status'
        result = self.manager.gitlab.http_get(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, dict)
        return result