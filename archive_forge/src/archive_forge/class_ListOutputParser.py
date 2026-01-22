from __future__ import annotations
import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
class ListOutputParser(BaseTransformOutputParser[List[str]]):
    """Parse the output of an LLM call to a list."""

    @property
    def _type(self) -> str:
        return 'list'

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call."""
        raise NotImplementedError

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[List[str]]:
        buffer = ''
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            buffer += chunk
            try:
                done_idx = 0
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        for part in self.parse(buffer):
            yield [part]

    async def _atransform(self, input: AsyncIterator[Union[str, BaseMessage]]) -> AsyncIterator[List[str]]:
        buffer = ''
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                chunk = chunk_content
            buffer += chunk
            try:
                done_idx = 0
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        for part in self.parse(buffer):
            yield [part]