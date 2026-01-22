import json
from typing import Generic, List, Type, TypeVar, Union
import pydantic  # pydantic: ignore
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
class PydanticOutputParser(JsonOutputParser, Generic[TBaseModel]):
    """Parse an output using a pydantic model."""
    pydantic_object: Type[TBaseModel]
    'The pydantic model to parse.'

    def _parse_obj(self, obj: dict) -> TBaseModel:
        if PYDANTIC_MAJOR_VERSION == 2:
            try:
                if issubclass(self.pydantic_object, pydantic.BaseModel):
                    return self.pydantic_object.model_validate(obj)
                elif issubclass(self.pydantic_object, pydantic.v1.BaseModel):
                    return self.pydantic_object.parse_obj(obj)
                else:
                    raise OutputParserException(f'Unsupported model version for PydanticOutputParser:                             {self.pydantic_object.__class__}')
            except (pydantic.ValidationError, pydantic.v1.ValidationError) as e:
                raise self._parser_exception(e, obj)
        else:
            try:
                return self.pydantic_object.parse_obj(obj)
            except pydantic.ValidationError as e:
                raise self._parser_exception(e, obj)

    def _parser_exception(self, e: Exception, json_object: dict) -> OutputParserException:
        json_string = json.dumps(json_object)
        name = self.pydantic_object.__name__
        msg = f'Failed to parse {name} from completion {json_string}. Got: {e}'
        return OutputParserException(msg, llm_output=json_string)

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> TBaseModel:
        json_object = super().parse_result(result)
        return self._parse_obj(json_object)

    def parse(self, text: str) -> TBaseModel:
        return super().parse(text)

    def get_format_instructions(self) -> str:
        schema = {k: v for k, v in self.pydantic_object.schema().items()}
        reduced_schema = schema
        if 'title' in reduced_schema:
            del reduced_schema['title']
        if 'type' in reduced_schema:
            del reduced_schema['type']
        schema_str = json.dumps(reduced_schema)
        return _PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return 'pydantic'

    @property
    def OutputType(self) -> Type[TBaseModel]:
        """Return the pydantic model."""
        return self.pydantic_object