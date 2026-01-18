import re
from typing import Callable, List, Optional, Set
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def unique_slug(slug: str, slugs: Set[str]) -> str:
    uniq = slug
    i = 1
    while uniq in slugs:
        uniq = f'{slug}-{i}'
        i += 1
    slugs.add(uniq)
    return uniq