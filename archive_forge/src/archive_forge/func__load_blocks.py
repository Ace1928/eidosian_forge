from typing import Any, Dict, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _load_blocks(self, block_id: str, num_tabs: int=0) -> str:
    """Read a block and its children."""
    result_lines_arr: List[str] = []
    cur_block_id: str = block_id
    while cur_block_id:
        data = self._request(BLOCK_URL.format(block_id=cur_block_id))
        for result in data['results']:
            result_obj = result[result['type']]
            if 'rich_text' not in result_obj:
                continue
            cur_result_text_arr: List[str] = []
            for rich_text in result_obj['rich_text']:
                if 'text' in rich_text:
                    cur_result_text_arr.append('\t' * num_tabs + rich_text['text']['content'])
            if result['has_children']:
                children_text = self._load_blocks(result['id'], num_tabs=num_tabs + 1)
                cur_result_text_arr.append(children_text)
            result_lines_arr.append('\n'.join(cur_result_text_arr))
        cur_block_id = data.get('next_cursor')
    return '\n'.join(result_lines_arr)