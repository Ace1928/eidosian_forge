from __future__ import annotations
from typing import Any
from .base_comm import BaseComm, BuffersType, CommManager, MaybeDict
def _get_comm_manager() -> CommManager:
    """Get the current Comm manager, creates one if there is none.

    This method is intended to be replaced if needed (if you want to manage multiple CommManagers).
    """
    global _comm_manager
    if _comm_manager is None:
        _comm_manager = CommManager()
    return _comm_manager