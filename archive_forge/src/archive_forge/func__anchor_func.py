import re
from typing import Callable, List, Optional, Set
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def _anchor_func(state: StateCore) -> None:
    slugs: Set[str] = set()
    for idx, token in enumerate(state.tokens):
        if token.type != 'heading_open':
            continue
        level = int(token.tag[1])
        if level not in selected_levels:
            continue
        inline_token = state.tokens[idx + 1]
        assert inline_token.children is not None
        title = ''.join((child.content for child in inline_token.children if child.type in ['text', 'code_inline']))
        slug = unique_slug(slug_func(title), slugs)
        token.attrSet('id', slug)
        if permalink:
            link_open = Token('link_open', 'a', 1)
            link_open.attrSet('class', 'header-anchor')
            link_open.attrSet('href', f'#{slug}')
            link_tokens = [link_open, Token('html_block', '', 0, content=permalinkSymbol), Token('link_close', 'a', -1)]
            if permalinkBefore:
                inline_token.children = link_tokens + ([Token('text', '', 0, content=' ')] if permalinkSpace else []) + inline_token.children
            else:
                inline_token.children.extend(([Token('text', '', 0, content=' ')] if permalinkSpace else []) + link_tokens)