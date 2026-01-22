import base64
from typing import (
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectFile(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'file_path'
    _repr_attr = 'file_path'
    branch: str
    commit_message: str
    file_path: str
    manager: 'ProjectFileManager'

    def decode(self) -> bytes:
        """Returns the decoded content of the file.

        Returns:
            The decoded content.
        """
        return base64.b64decode(self.content)

    def save(self, branch: str, commit_message: str, **kwargs: Any) -> None:
        """Save the changes made to the file to the server.

        The object is updated to match what the server returns.

        Args:
            branch: Branch in which the file will be updated
            commit_message: Message to send with the commit
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabUpdateError: If the server cannot perform the request
        """
        self.branch = branch
        self.commit_message = commit_message
        self.file_path = utils.EncodedId(self.file_path)
        super().save(**kwargs)

    @exc.on_http_error(exc.GitlabDeleteError)
    def delete(self, branch: str, commit_message: str, **kwargs: Any) -> None:
        """Delete the file from the server.

        Args:
            branch: Branch from which the file will be removed
            commit_message: Commit message for the deletion
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabDeleteError: If the server cannot perform the request
        """
        file_path = self.encoded_id
        if TYPE_CHECKING:
            assert isinstance(file_path, str)
        self.manager.delete(file_path, branch, commit_message, **kwargs)