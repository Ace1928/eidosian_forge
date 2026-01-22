import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _osc_link(ansi: str) -> Iterator[Union[str, OSC_Link]]:
    last_end = 0
    for match in self.osc_link_re.finditer(ansi):
        trailer = ansi[last_end:match.start()]
        yield trailer
        url = match.groups()[0]
        text = match.groups()[1]
        yield OSC_Link(url, text)
        last_end = match.end()
    yield ansi[last_end:]