from typing import Optional
from ..._models import BaseModel
from .function_parameters import FunctionParameters
The parameters the functions accepts, described as a JSON Schema object.

    See the
    [guide](https://platform.openai.com/docs/guides/text-generation/function-calling)
    for examples, and the
    [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for
    documentation about the format.

    Omitting `parameters` defines a function with an empty parameter list.
    