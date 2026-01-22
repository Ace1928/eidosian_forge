import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
def add_content_to_text(text: str, content: str, add_after: Optional[Union[str, Pattern]]=None, add_before: Optional[Union[str, Pattern]]=None, exact_match: bool=False) -> str:
    """
    A utility to add some content inside a given text.

    Args:
       text (`str`): The text in which we want to insert some content.
       content (`str`): The content to add.
       add_after (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added after the first instance matching it.
       add_before (`str` or `Pattern`):
           The pattern to test on a line of `text`, the new content is added before the first instance matching it.
       exact_match (`bool`, *optional*, defaults to `False`):
           A line is considered a match with `add_after` or `add_before` if it matches exactly when `exact_match=True`,
           otherwise, if `add_after`/`add_before` is present in the line.

    <Tip warning={true}>

    The arguments `add_after` and `add_before` are mutually exclusive, and one exactly needs to be provided.

    </Tip>

    Returns:
        `str`: The text with the new content added if a match was found.
    """
    if add_after is None and add_before is None:
        raise ValueError('You need to pass either `add_after` or `add_before`')
    if add_after is not None and add_before is not None:
        raise ValueError("You can't pass both `add_after` or `add_before`")
    pattern = add_after if add_before is None else add_before

    def this_is_the_line(line):
        if isinstance(pattern, Pattern):
            return pattern.search(line) is not None
        elif exact_match:
            return pattern == line
        else:
            return pattern in line
    new_lines = []
    for line in text.split('\n'):
        if this_is_the_line(line):
            if add_before is not None:
                new_lines.append(content)
            new_lines.append(line)
            if add_after is not None:
                new_lines.append(content)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)