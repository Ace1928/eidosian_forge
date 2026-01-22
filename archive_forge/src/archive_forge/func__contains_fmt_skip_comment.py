import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Collection, Final, Iterator, List, Optional, Tuple, Union
from black.mode import Mode, Preview
from black.nodes import (
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def _contains_fmt_skip_comment(comment_line: str, mode: Mode) -> bool:
    """
    Checks if the given comment contains FMT_SKIP alone or paired with other comments.
    Matching styles:
      # fmt:skip                           <-- single comment
      # noqa:XXX # fmt:skip # a nice line  <-- multiple comments (Preview)
      # pylint:XXX; fmt:skip               <-- list of comments (; separated, Preview)
    """
    semantic_comment_blocks = [comment_line, *[_COMMENT_PREFIX + comment.strip() for comment in comment_line.split(_COMMENT_PREFIX)[1:]], *[_COMMENT_PREFIX + comment.strip() for comment in comment_line.strip(_COMMENT_PREFIX).split(_COMMENT_LIST_SEPARATOR)]]
    return any((comment in FMT_SKIP for comment in semantic_comment_blocks))