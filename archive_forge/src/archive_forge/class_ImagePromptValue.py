from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Literal, Sequence, cast
from typing_extensions import TypedDict
from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
class ImagePromptValue(PromptValue):
    """Image prompt value."""
    image_url: ImageURL
    'Prompt image.'
    type: Literal['ImagePromptValue'] = 'ImagePromptValue'

    def to_string(self) -> str:
        """Return prompt as string."""
        return self.image_url['url']

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return [HumanMessage(content=[cast(dict, self.image_url)])]