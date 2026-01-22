from typing import Any, List, Literal
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """Function Message chunk."""
    type: Literal['FunctionMessageChunk'] = 'FunctionMessageChunk'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']

    def __add__(self, other: Any) -> BaseMessageChunk:
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                raise ValueError('Cannot concatenate FunctionMessageChunks with different names.')
            return self.__class__(name=self.name, content=merge_content(self.content, other.content), additional_kwargs=merge_dicts(self.additional_kwargs, other.additional_kwargs), response_metadata=merge_dicts(self.response_metadata, other.response_metadata), id=self.id)
        return super().__add__(other)