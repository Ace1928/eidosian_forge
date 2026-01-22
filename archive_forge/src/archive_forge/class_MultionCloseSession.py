from typing import TYPE_CHECKING, Optional, Type
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
class MultionCloseSession(BaseTool):
    """Tool that closes an existing Multion Browser Window with provided fields.

    Attributes:
        name: The name of the tool. Default: "close_multion_session"
        description: The description of the tool.
        args_schema: The schema for the tool's arguments. Default: UpdateSessionSchema
    """
    name: str = 'close_multion_session'
    description: str = 'Use this tool to close an existing corresponding Multion Browser Window with provided fields. Note: SessionId must be received from previous Browser window creation.'
    args_schema: Type[CloseSessionSchema] = CloseSessionSchema
    sessionId: str = ''

    def _run(self, sessionId: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> None:
        try:
            try:
                multion.close_session(sessionId)
            except Exception as e:
                print(f'{e}, retrying...')
        except Exception as e:
            raise Exception(f'An error occurred: {e}')