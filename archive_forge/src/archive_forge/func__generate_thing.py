import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def _generate_thing(x):
    if isinstance(x, internal.Paragraph):
        return _internal_children_to_text(x.children)
    elif isinstance(x, internal.Text):
        if x.inline_code:
            return InlineCode(x.text)
        elif x.inline_comments:
            return TextWithInlineComments(text=x.text, _inline_comments=x.inline_comments)
        return x.text
    elif isinstance(x, internal.InlineLink):
        text_obj = x.children[0]
        if text_obj.inline_comments:
            text = TextWithInlineComments(text=text_obj.text, _inline_comments=text_obj.inline_comments)
        else:
            text = text_obj.text
        return Link(url=x.url, text=text)
    elif isinstance(x, internal.InlineLatex):
        return InlineLatex(text=x.content)