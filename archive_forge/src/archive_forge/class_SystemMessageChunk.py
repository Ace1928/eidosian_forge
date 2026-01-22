from typing import List, Literal
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """System Message chunk."""
    type: Literal['SystemMessageChunk'] = 'SystemMessageChunk'

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'messages']