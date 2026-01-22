import string
from typing import Callable, List
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
def _word_count_rule(state: StateCore) -> None:
    text: List[str] = []
    words = 0
    for token in state.tokens:
        if token.type == 'text':
            words += count_func(token.content)
            if store_text:
                text.append(token.content)
        elif token.type == 'inline':
            for child in token.children or ():
                if child.type == 'text':
                    words += count_func(child.content)
                    if store_text:
                        text.append(child.content)
    data = state.env.setdefault('wordcount', {})
    if store_text:
        data.setdefault('text', [])
        data['text'] += text
    data.setdefault('words', 0)
    data['words'] += words
    data['minutes'] = int(round(data['words'] / per_minute))