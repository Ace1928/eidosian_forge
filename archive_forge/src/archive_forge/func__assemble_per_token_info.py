import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from ..errors import Errors
from ..util import escape_html, minify_html, registry
from .templates import (
@staticmethod
def _assemble_per_token_info(tokens: List[str], spans: List[Dict[str, Any]]) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Assembles token info used to generate markup in render_spans().
        tokens (List[str]): Tokens in text.
        spans (List[Dict[str, Any]]): Spans in text.
        RETURNS (List[Dict[str, List[Dict, str, Any]]]): Per token info needed to render HTML markup for given tokens
            and spans.
        """
    per_token_info: List[Dict[str, List[Dict[str, Any]]]] = []
    spans = sorted(spans, key=lambda s: (s['start_token'], -(s['end_token'] - s['start_token']), s['label']))
    for s in spans:
        s['render_slot'] = 0
    for idx, token in enumerate(tokens):
        token_markup: Dict[str, Any] = {}
        token_markup['text'] = token
        intersecting_spans: List[Dict[str, Any]] = []
        entities = []
        for span in spans:
            ent = {}
            if span['start_token'] <= idx < span['end_token']:
                span_start = idx == span['start_token']
                ent['label'] = span['label']
                ent['is_start'] = span_start
                if span_start:
                    span['render_slot'] = (intersecting_spans[-1]['render_slot'] if len(intersecting_spans) else 0) + 1
                intersecting_spans.append(span)
                ent['render_slot'] = span['render_slot']
                kb_id = span.get('kb_id', '')
                kb_url = span.get('kb_url', '#')
                ent['kb_link'] = TPL_KB_LINK.format(kb_id=kb_id, kb_url=kb_url) if kb_id else ''
                entities.append(ent)
            else:
                span['render_slot'] = 0
        token_markup['entities'] = entities
        per_token_info.append(token_markup)
    return per_token_info