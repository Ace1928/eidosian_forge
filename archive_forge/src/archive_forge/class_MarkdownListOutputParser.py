from __future__ import annotations
import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
class MarkdownListOutputParser(ListOutputParser):
    """Parse a markdown list."""
    pattern = '^\\s*[-*]\\s([^\\n]+)$'

    def get_format_instructions(self) -> str:
        return 'Your response should be a markdown list, eg: `- foo\n- bar\n- baz`'

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return re.findall(self.pattern, text, re.MULTILINE)

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call."""
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return 'markdown-list'