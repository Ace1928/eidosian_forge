from __future__ import annotations
import gc
import niquests
from urllib.parse import urljoin
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
class BaseEvent(BaseModel):
    """
    Base Event
    """
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    @property
    def is_valid(self) -> bool:
        """
        Returns True if the event is valid
        """
        return True

    def prepare_request(self, exclude_none: Optional[bool]=True, batched: Optional[bool]=False, **kwargs) -> Dict[str, Any]:
        """
        Formats the event for batching
        """
        return self.model_dump(exclude_none=exclude_none, **kwargs)