from __future__ import annotations
import time
from ..types.properties import StatefulProperty
from ..types.auth import APIKeyData
from .admin import AZManagementClient, logger
from ..utils.lazy import get_az_flow
from typing import List, Optional, Any, TYPE_CHECKING
@property
def user_data_flow(self) -> 'UserDataFlow':
    """
        Returns the User Data Flow
        """
    if self._user_data_flow is None:
        self._user_data_flow = get_az_flow('user_data', user_id=self.user_id)
    return self._user_data_flow