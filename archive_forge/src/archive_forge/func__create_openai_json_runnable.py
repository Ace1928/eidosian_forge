import json
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Type, Union
from langchain_core.output_parsers import (
from langchain_core.output_parsers.openai_functions import (
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import (
from langchain.output_parsers import (
def _create_openai_json_runnable(output_schema: Union[Dict[str, Any], Type[BaseModel]], llm: Runnable, prompt: Optional[BasePromptTemplate]=None, *, output_parser: Optional[Union[BaseOutputParser, BaseGenerationOutputParser]]=None) -> Runnable:
    """"""
    if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
        output_parser = output_parser or PydanticOutputParser(pydantic_object=output_schema)
        schema_as_dict = convert_to_openai_function(output_schema)['parameters']
    else:
        output_parser = output_parser or JsonOutputParser()
        schema_as_dict = output_schema
    llm = llm.bind(response_format={'type': 'json_object'})
    if prompt:
        if 'output_schema' in prompt.input_variables:
            prompt = prompt.partial(output_schema=json.dumps(schema_as_dict, indent=2))
        return prompt | llm | output_parser
    else:
        return llm | output_parser