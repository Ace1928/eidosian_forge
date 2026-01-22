from typing import Optional
from typing_extensions import Literal
from ....._models import BaseModel
class CodeInterpreterLogs(BaseModel):
    index: int
    'The index of the output in the outputs array.'
    type: Literal['logs']
    'Always `logs`.'
    logs: Optional[str] = None
    'The text output from the Code Interpreter tool call.'