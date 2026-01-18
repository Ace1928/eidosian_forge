import re
from copy import deepcopy
from typing import Any, Callable, List, Match, Optional, Pattern, Tuple, Union
from docutils import nodes
from docutils.nodes import TextElement
from sphinx import addnodes
from sphinx.config import Config
from sphinx.util import logging
def verify_description_mode(mode: str) -> None:
    if mode not in ('lastIsName', 'noneIsName', 'markType', 'markName', 'param', 'udl'):
        raise Exception("Description mode '%s' is invalid." % mode)