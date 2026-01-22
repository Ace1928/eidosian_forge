import html
import json
import pickle
import re
import warnings
from importlib import import_module
from os import path
from typing import IO, Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, Node
from sphinx import addnodes, package_dir
from sphinx.deprecation import RemovedInSphinx70Warning
from sphinx.environment import BuildEnvironment
from sphinx.util import split_into
from sphinx.search.en import SearchEnglish
class _JavaScriptIndex:
    """
    The search index as JavaScript file that calls a function
    on the documentation search object to register the index.
    """
    PREFIX = 'Search.setIndex('
    SUFFIX = ')'

    def dumps(self, data: Any) -> str:
        return self.PREFIX + json.dumps(data) + self.SUFFIX

    def loads(self, s: str) -> Any:
        data = s[len(self.PREFIX):-len(self.SUFFIX)]
        if not data or not s.startswith(self.PREFIX) or (not s.endswith(self.SUFFIX)):
            raise ValueError('invalid data')
        return json.loads(data)

    def dump(self, data: Any, f: IO) -> None:
        f.write(self.dumps(data))

    def load(self, f: IO) -> Any:
        return self.loads(f.read())