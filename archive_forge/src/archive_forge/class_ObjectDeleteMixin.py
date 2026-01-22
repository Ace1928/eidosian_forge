import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
class ObjectDeleteMixin(_RestObjectBase):
    """Mixin for RESTObject's that can be deleted."""
    _id_attr: Optional[str]
    _attrs: Dict[str, Any]
    _module: ModuleType
    _parent_attrs: Dict[str, Any]
    _updated_attrs: Dict[str, Any]
    manager: base.RESTManager

    def delete(self, **kwargs: Any) -> None:
        """Delete the object from the server.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server cannot perform the request
        """
        if TYPE_CHECKING:
            assert isinstance(self.manager, DeleteMixin)
            assert self.encoded_id is not None
        self.manager.delete(self.encoded_id, **kwargs)