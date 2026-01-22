from __future__ import annotations
from typing import Any, Dict, List, Tuple, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
class MarkdownHeaderTextSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(self, headers_to_split_on: List[Tuple[str, str]], return_each_line: bool=False, strip_headers: bool=True):
        """Create a new MarkdownHeaderTextSplitter.

        Args:
            headers_to_split_on: Headers we want to track
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
        """
        self.return_each_line = return_each_line
        self.headers_to_split_on = sorted(headers_to_split_on, key=lambda split: len(split[0]), reverse=True)
        self.strip_headers = strip_headers

    def aggregate_lines_to_chunks(self, lines: List[LineType]) -> List[Document]:
        """Combine lines with common metadata into chunks
        Args:
            lines: Line of text / associated header metadata
        """
        aggregated_chunks: List[LineType] = []
        for line in lines:
            if aggregated_chunks and aggregated_chunks[-1]['metadata'] == line['metadata']:
                aggregated_chunks[-1]['content'] += '  \n' + line['content']
            elif aggregated_chunks and aggregated_chunks[-1]['metadata'] != line['metadata'] and (len(aggregated_chunks[-1]['metadata']) < len(line['metadata'])) and (aggregated_chunks[-1]['content'].split('\n')[-1][0] == '#') and (not self.strip_headers):
                aggregated_chunks[-1]['content'] += '  \n' + line['content']
                aggregated_chunks[-1]['metadata'] = line['metadata']
            else:
                aggregated_chunks.append(line)
        return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in aggregated_chunks]

    def split_text(self, text: str) -> List[Document]:
        """Split markdown file
        Args:
            text: Markdown file"""
        lines = text.split('\n')
        lines_with_metadata: List[LineType] = []
        current_content: List[str] = []
        current_metadata: Dict[str, str] = {}
        header_stack: List[HeaderType] = []
        initial_metadata: Dict[str, str] = {}
        in_code_block = False
        opening_fence = ''
        for line in lines:
            stripped_line = line.strip()
            if not in_code_block:
                if stripped_line.startswith('```') and stripped_line.count('```') == 1:
                    in_code_block = True
                    opening_fence = '```'
                elif stripped_line.startswith('~~~'):
                    in_code_block = True
                    opening_fence = '~~~'
            elif stripped_line.startswith(opening_fence):
                in_code_block = False
                opening_fence = ''
            if in_code_block:
                current_content.append(stripped_line)
                continue
            for sep, name in self.headers_to_split_on:
                if stripped_line.startswith(sep) and (len(stripped_line) == len(sep) or stripped_line[len(sep)] == ' '):
                    if name is not None:
                        current_header_level = sep.count('#')
                        while header_stack and header_stack[-1]['level'] >= current_header_level:
                            popped_header = header_stack.pop()
                            if popped_header['name'] in initial_metadata:
                                initial_metadata.pop(popped_header['name'])
                        header: HeaderType = {'level': current_header_level, 'name': name, 'data': stripped_line[len(sep):].strip()}
                        header_stack.append(header)
                        initial_metadata[name] = header['data']
                    if current_content:
                        lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                        current_content.clear()
                    if not self.strip_headers:
                        current_content.append(stripped_line)
                    break
            else:
                if stripped_line:
                    current_content.append(stripped_line)
                elif current_content:
                    lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata.copy()})
                    current_content.clear()
            current_metadata = initial_metadata.copy()
        if current_content:
            lines_with_metadata.append({'content': '\n'.join(current_content), 'metadata': current_metadata})
        if not self.return_each_line:
            return self.aggregate_lines_to_chunks(lines_with_metadata)
        else:
            return [Document(page_content=chunk['content'], metadata=chunk['metadata']) for chunk in lines_with_metadata]