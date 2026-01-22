import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def append_unless_default(output: List[str], value: int, default: int) -> None:
    if value != default:
        css_class = 'ansi%d' % value
        output.append(css_class)