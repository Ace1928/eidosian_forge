from inspect import Parameter, Signature, getsource
from typing import Any, Dict, List, cast
import sphinx
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.pycode.ast import unparse as ast_unparse
from sphinx.util import inspect, logging
Update annotations info of *obj* using type_comments.