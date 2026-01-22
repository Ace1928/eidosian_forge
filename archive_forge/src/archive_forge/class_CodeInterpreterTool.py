from typing_extensions import Literal
from ..._models import BaseModel
class CodeInterpreterTool(BaseModel):
    type: Literal['code_interpreter']
    'The type of tool being defined: `code_interpreter`'