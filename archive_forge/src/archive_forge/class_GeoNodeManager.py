from typing import Any, cast, Dict, List, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GeoNodeManager(RetrieveMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/geo_nodes'
    _obj_cls = GeoNode
    _update_attrs = RequiredOptional(optional=('enabled', 'url', 'files_max_capacity', 'repos_max_capacity'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GeoNode:
        return cast(GeoNode, super().get(id=id, lazy=lazy, **kwargs))

    @cli.register_custom_action('GeoNodeManager')
    @exc.on_http_error(exc.GitlabGetError)
    def status(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Get the status of all the geo nodes.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The status of all the geo nodes
        """
        result = self.gitlab.http_list('/geo_nodes/status', **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, list)
        return result

    @cli.register_custom_action('GeoNodeManager')
    @exc.on_http_error(exc.GitlabGetError)
    def current_failures(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Get the list of failures on the current geo node.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabGetError: If the server failed to perform the request

        Returns:
            The list of failures
        """
        result = self.gitlab.http_list('/geo_nodes/current/failures', **kwargs)
        if TYPE_CHECKING:
            assert isinstance(result, list)
        return result