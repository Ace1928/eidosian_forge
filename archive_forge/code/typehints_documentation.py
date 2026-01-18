import re
from collections import OrderedDict
from typing import Any, Dict, Iterable, Set, cast
from docutils import nodes
from docutils.nodes import Element
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect, typing
Record type hints to env object.