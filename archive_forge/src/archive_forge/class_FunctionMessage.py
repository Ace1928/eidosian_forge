from typing import Any, List, Literal
from langchain_core.messages.base import (
from langchain_core.utils._merge import merge_dicts
class FunctionMessage(BaseMessage):
    """Message for passing the result of executing a function back to a model."""
    name: str
    'The name of the function that was executed.'
    type: Literal['function'] = 'function'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']