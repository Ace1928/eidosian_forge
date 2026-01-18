from __future__ import annotations
import re
from uuid import uuid4
from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token
def todoify(token: Token) -> None:
    assert token.children is not None
    token.children.insert(0, make_checkbox(token))
    token.children[1].content = token.children[1].content[3:]
    token.content = token.content[3:]
    if use_label_wrapper:
        if use_label_after:
            token.children.pop()
            checklist_id = f'task-item-{uuid4()}'
            token.children[0].content = token.children[0].content[0:-1] + f' id="{checklist_id}">'
            token.children.append(after_label(token.content, checklist_id))
        else:
            token.children.insert(0, begin_label())
            token.children.append(end_label())