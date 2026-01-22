from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadMessageIncomplete(BaseModel):
    data: Message
    '\n    Represents a message within a\n    [thread](https://platform.openai.com/docs/api-reference/threads).\n    '
    event: Literal['thread.message.incomplete']