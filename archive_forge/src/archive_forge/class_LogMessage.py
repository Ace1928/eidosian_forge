from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class LogMessage(BaseMessage):
    msg: Literal[ServerMessage.log] = ServerMessage.log
    log: str
    level: Literal['info', 'warning']