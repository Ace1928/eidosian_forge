from typing_extensions import Literal
from ...._models import BaseModel
class RequiredActionFunctionToolCall(BaseModel):
    id: str
    'The ID of the tool call.\n\n    This ID must be referenced when you submit the tool outputs in using the\n    [Submit tool outputs to run](https://platform.openai.com/docs/api-reference/runs/submitToolOutputs)\n    endpoint.\n    '
    function: Function
    'The function definition.'
    type: Literal['function']
    'The type of tool call the output is required for.\n\n    For now, this is always `function`.\n    '