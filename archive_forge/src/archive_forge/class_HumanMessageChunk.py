from typing import List, Literal
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """Human Message chunk."""
    type: Literal['HumanMessageChunk'] = 'HumanMessageChunk'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']