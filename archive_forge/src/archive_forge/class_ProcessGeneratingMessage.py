from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class ProcessGeneratingMessage(BaseMessage):
    msg: Literal[ServerMessage.process_generating] = ServerMessage.process_generating
    output: dict
    success: bool