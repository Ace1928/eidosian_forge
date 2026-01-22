from typing import Any, Dict, List, Optional
import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env
from typing_extensions import Literal
def _parse_snippets(self, results: dict) -> List[str]:
    snippets = []
    if results.get('answerBox'):
        answer_box = results.get('answerBox', {})
        if answer_box.get('answer'):
            return [answer_box.get('answer')]
        elif answer_box.get('snippet'):
            return [answer_box.get('snippet').replace('\n', ' ')]
        elif answer_box.get('snippetHighlighted'):
            return answer_box.get('snippetHighlighted')
    if results.get('knowledgeGraph'):
        kg = results.get('knowledgeGraph', {})
        title = kg.get('title')
        entity_type = kg.get('type')
        if entity_type:
            snippets.append(f'{title}: {entity_type}.')
        description = kg.get('description')
        if description:
            snippets.append(description)
        for attribute, value in kg.get('attributes', {}).items():
            snippets.append(f'{title} {attribute}: {value}.')
    for result in results[self.result_key_for_type[self.type]][:self.k]:
        if 'snippet' in result:
            snippets.append(result['snippet'])
        for attribute, value in result.get('attributes', {}).items():
            snippets.append(f'{attribute}: {value}.')
    if len(snippets) == 0:
        return ['No good Google Search Result was found']
    return snippets