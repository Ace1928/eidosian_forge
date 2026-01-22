from typing import Union
from typing_extensions import Literal, Annotated
from .thread import Thread
from ..shared import ErrorObject
from .threads import Run, Message, MessageDeltaEvent
from ..._utils import PropertyInfo
from ..._models import BaseModel
from .threads.runs import RunStep, RunStepDeltaEvent
class ThreadRunExpired(BaseModel):
    data: Run
    '\n    Represents an execution run on a\n    [thread](https://platform.openai.com/docs/api-reference/threads).\n    '
    event: Literal['thread.run.expired']