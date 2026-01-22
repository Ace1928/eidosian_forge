import ast
import json
import os
import sys
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
def _parse_property_kwargs(keywords: List[ast.keyword], prop: PropertyEntry):
    """Parse keyword arguments of @Property"""
    for k in keywords:
        if k.arg == 'notify':
            prop['notify'] = _name(k.value)