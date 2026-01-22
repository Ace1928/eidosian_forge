import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def _parse_article(self, uid: str, text_dict: dict) -> dict:
    try:
        ar = text_dict['PubmedArticleSet']['PubmedArticle']['MedlineCitation']['Article']
    except KeyError:
        ar = text_dict['PubmedArticleSet']['PubmedBookArticle']['BookDocument']
    abstract_text = ar.get('Abstract', {}).get('AbstractText', [])
    summaries = [f'{txt['@Label']}: {txt['#text']}' for txt in abstract_text if '#text' in txt and '@Label' in txt]
    summary = '\n'.join(summaries) if summaries else abstract_text if isinstance(abstract_text, str) else '\n'.join((str(value) for value in abstract_text.values())) if isinstance(abstract_text, dict) else 'No abstract available'
    a_d = ar.get('ArticleDate', {})
    pub_date = '-'.join([a_d.get('Year', ''), a_d.get('Month', ''), a_d.get('Day', '')])
    return {'uid': uid, 'Title': ar.get('ArticleTitle', ''), 'Published': pub_date, 'Copyright Information': ar.get('Abstract', {}).get('CopyrightInformation', ''), 'Summary': summary}