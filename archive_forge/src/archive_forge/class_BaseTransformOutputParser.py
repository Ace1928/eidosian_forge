from __future__ import annotations
from typing import (
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.output_parsers.base import BaseOutputParser, T
from langchain_core.outputs import (
class BaseTransformOutputParser(BaseOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    def transform(self, input: Iterator[Union[str, BaseMessage]], config: Optional[RunnableConfig]=None, **kwargs: Any) -> Iterator[T]:
        yield from self._transform_stream_with_config(input, self._transform, config, run_type='parser')

    async def atransform(self, input: AsyncIterator[Union[str, BaseMessage]], config: Optional[RunnableConfig]=None, **kwargs: Any) -> AsyncIterator[T]:
        async for chunk in self._atransform_stream_with_config(input, self._atransform, config, run_type='parser'):
            yield chunk