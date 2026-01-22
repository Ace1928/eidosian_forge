import logging
import re
from argparse import (
from collections import defaultdict
from functools import total_ordering
from itertools import starmap
from string import Template
from typing import Any, Dict, List
from typing import Optional as Opt
from typing import Union
class Required:
    """Example: `ArgumentParser.add_argument(..., choices=Required.FILE)`."""
    FILE = [Choice('file', True)]
    DIR = DIRECTORY = [Choice('directory', True)]