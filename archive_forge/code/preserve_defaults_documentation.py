import ast
import inspect
import sys
from inspect import Parameter
from typing import Any, Dict, List, Optional
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import logging
Update defvalue info of *obj* using type_comments.