import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
class OSC_Link:

    def __init__(self, url: str, text: str) -> None:
        self.url = url
        self.text = text