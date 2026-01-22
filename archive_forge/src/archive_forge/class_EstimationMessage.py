from typing import List, Literal, Optional, Union
from gradio_client.utils import ServerMessage
from gradio.data_classes import BaseModel
class EstimationMessage(BaseMessage):
    msg: Literal[ServerMessage.estimation] = ServerMessage.estimation
    rank: Optional[int] = None
    queue_size: int
    rank_eta: Optional[float] = None