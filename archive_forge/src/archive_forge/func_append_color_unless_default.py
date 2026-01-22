import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def append_color_unless_default(output: List[str], color: Tuple[int, Optional[str]], default: int, negative: bool, neg_css_class: str) -> None:
    value, parameter = color
    if value != default:
        prefix = 'inv' if negative else 'ansi'
        css_class_index = str(value) if parameter is None else '%d-%s' % (value, parameter)
        output.append(prefix + css_class_index)
    elif negative:
        output.append(neg_css_class)