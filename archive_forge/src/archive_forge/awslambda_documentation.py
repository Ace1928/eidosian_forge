import json
from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator

        Invokes the lambda function and returns the
        result.

        Args:
            query: an input to passed to the lambda
                function as the ``body`` of a JSON
                object.
        