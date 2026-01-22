from __future__ import annotations
from typing import Any, Dict, List, Tuple, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
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