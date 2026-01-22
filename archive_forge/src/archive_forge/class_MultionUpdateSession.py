from typing import TYPE_CHECKING, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class MultionUpdateSession(BaseTool):
    """Tool that updates an existing Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "update_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments. Default: UpdateSessionSchema
    """
    name: str = 'update_multion_session'
    description: str = 'Use this tool to update an existing corresponding Multion Browser Window with provided fields. Note: sessionId must be received from previous Browser window creation.'
    args_schema: Type[UpdateSessionSchema] = UpdateSessionSchema
    sessionId: str = ''

    def _run(self, sessionId: str, query: str, url: Optional[str]='https://www.google.com/', run_manager: Optional[CallbackManagerForToolRun]=None) -> dict:
        try:
            try:
                response = multion.update_session(sessionId, {'input': query, 'url': url})
                content = {'sessionId': sessionId, 'Response': response['message']}
                self.sessionId = sessionId
                return content
            except Exception as e:
                print(f'{e}, retrying...')
                return {'error': f'{e}', 'Response': 'retrying...'}
        except Exception as e:
            raise Exception(f'An error occurred: {e}')