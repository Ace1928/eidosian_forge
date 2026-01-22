from __future__ import annotations
import functools
import logging
from collections import defaultdict
from typing import (
from langsmith import run_helpers
def _reduce_completions(all_chunks: List[Completion]) -> dict:
    all_content = []
    for chunk in all_chunks:
        content = chunk.choices[0].text
        if content is not None:
            all_content.append(content)
    content = ''.join(all_content)
    if all_chunks:
        d = all_chunks[-1].model_dump()
        d['choices'] = [{'text': content}]
    else:
        d = {'choices': [{'text': content}]}
    return d