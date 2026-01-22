from __future__ import annotations
import json
from json import JSONDecodeError
from typing import Any, List, Optional, Type, TypeVar, Union
import jsonpatch  # type: ignore[import]
import pydantic  # pydantic: ignore
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION
class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations
    describing the difference between the previous and the current object.
    """
    pydantic_object: Optional[Type[TBaseModel]] = None

    def _diff(self, prev: Optional[Any], next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def _get_schema(self, pydantic_object: Type[TBaseModel]) -> dict[str, Any]:
        if PYDANTIC_MAJOR_VERSION == 2:
            if issubclass(pydantic_object, pydantic.BaseModel):
                return pydantic_object.model_json_schema()
            elif issubclass(pydantic_object, pydantic.v1.BaseModel):
                return pydantic_object.schema()
        return pydantic_object.schema()

    def parse_result(self, result: List[Generation], *, partial: bool=False) -> Any:
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f'Invalid json output: {text}'
                raise OutputParserException(msg, llm_output=text) from e

    def parse(self, text: str) -> Any:
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        if self.pydantic_object is None:
            return 'Return a JSON object.'
        else:
            schema = {k: v for k, v in self._get_schema(self.pydantic_object).items()}
            reduced_schema = schema
            if 'title' in reduced_schema:
                del reduced_schema['title']
            if 'type' in reduced_schema:
                del reduced_schema['type']
            schema_str = json.dumps(reduced_schema)
            return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return 'simple_json_output_parser'