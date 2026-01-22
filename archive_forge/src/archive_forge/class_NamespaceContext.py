from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
class NamespaceContext(TreeContextMixin, ValueContext):

    def get_filters(self, until_position=None, origin_scope=None):
        return self._value.get_filters()

    def get_value(self):
        return self._value

    @property
    def string_names(self):
        return self._value.string_names

    def py__file__(self) -> Optional[Path]:
        return self._value.py__file__()