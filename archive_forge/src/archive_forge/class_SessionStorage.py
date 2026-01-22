from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Protocol, cast
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.script_data import ScriptData
from streamlit.runtime.scriptrunner.script_cache import ScriptCache
from streamlit.runtime.uploaded_file_manager import UploadedFileManager
class SessionStorage(Protocol):

    @abstractmethod
    def get(self, session_id: str) -> SessionInfo | None:
        """Return the SessionInfo corresponding to session_id, or None if one does not
        exist.

        Parameters
        ----------
        session_id
            The unique ID of the session being fetched.

        Returns
        -------
        SessionInfo or None

        Raises
        ------
        SessionStorageError
            Raised if an error occurs while attempting to fetch the session. This will
            generally happen if there is an error with the underlying storage backend
            (e.g. if we lose our connection to it).
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, session_info: SessionInfo) -> None:
        """Save the given session.

        Parameters
        ----------
        session_info
            The SessionInfo being saved.

        Raises
        ------
        SessionStorageError
            Raised if an error occurs while saving the given session.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Mark the session corresponding to session_id for deletion and stop tracking
        it.

        Note that:
          * Calling delete on an ID corresponding to a nonexistent session is a no-op.
          * Calling delete on an ID should cause the given session to no longer be
            tracked by this SessionStorage, but exactly when and how the session's data
            is eventually cleaned up is a detail left up to the implementation.

        Parameters
        ----------
        session_id
            The unique ID of the session to delete.

        Raises
        ------
        SessionStorageError
            Raised if an error occurs while attempting to delete the session.
        """
        raise NotImplementedError

    @abstractmethod
    def list(self) -> list[SessionInfo]:
        """List all sessions tracked by this SessionStorage.

        Returns
        -------
        List[SessionInfo]

        Raises
        ------
        SessionStorageError
            Raised if an error occurs while attempting to list sessions.
        """
        raise NotImplementedError