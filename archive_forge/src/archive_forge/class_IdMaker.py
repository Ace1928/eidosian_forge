import abc
from collections import Counter
from collections import defaultdict
import dataclasses
import enum
import fnmatch
from functools import partial
import inspect
import itertools
import os
from pathlib import Path
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import _pytest
from _pytest import fixtures
from _pytest import nodes
from _pytest._code import filter_traceback
from _pytest._code import getfslineno
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import TerminalRepr
from _pytest._code.code import Traceback
from _pytest._io import TerminalWriter
from _pytest._io.saferepr import saferepr
from _pytest.compat import ascii_escaped
from _pytest.compat import get_default_arg_names
from _pytest.compat import get_real_func
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_async_function
from _pytest.compat import is_generator
from _pytest.compat import LEGACY_PATH
from _pytest.compat import NOTSET
from _pytest.compat import safe_getattr
from _pytest.compat import safe_isclass
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import FixtureDef
from _pytest.fixtures import FixtureRequest
from _pytest.fixtures import FuncFixtureInfo
from _pytest.fixtures import get_scope_node
from _pytest.main import Session
from _pytest.mark import MARK_GEN
from _pytest.mark import ParameterSet
from _pytest.mark.structures import get_unpacked_marks
from _pytest.mark.structures import Mark
from _pytest.mark.structures import MarkDecorator
from _pytest.mark.structures import normalize_mark_list
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import scandir
from _pytest.scope import _ScopeName
from _pytest.scope import Scope
from _pytest.stash import StashKey
from _pytest.warning_types import PytestCollectionWarning
from _pytest.warning_types import PytestReturnNotNoneWarning
from _pytest.warning_types import PytestUnhandledCoroutineWarning
@final
@dataclasses.dataclass(frozen=True)
class IdMaker:
    """Make IDs for a parametrization."""
    __slots__ = ('argnames', 'parametersets', 'idfn', 'ids', 'config', 'nodeid', 'func_name')
    argnames: Sequence[str]
    parametersets: Sequence[ParameterSet]
    idfn: Optional[Callable[[Any], Optional[object]]]
    ids: Optional[Sequence[Optional[object]]]
    config: Optional[Config]
    nodeid: Optional[str]
    func_name: Optional[str]

    def make_unique_parameterset_ids(self) -> List[str]:
        """Make a unique identifier for each ParameterSet, that may be used to
        identify the parametrization in a node ID.

        Format is <prm_1_token>-...-<prm_n_token>[counter], where prm_x_token is
        - user-provided id, if given
        - else an id derived from the value, applicable for certain types
        - else <argname><parameterset index>
        The counter suffix is appended only in case a string wouldn't be unique
        otherwise.
        """
        resolved_ids = list(self._resolve_ids())
        if len(resolved_ids) != len(set(resolved_ids)):
            id_counts = Counter(resolved_ids)
            id_suffixes: Dict[str, int] = defaultdict(int)
            for index, id in enumerate(resolved_ids):
                if id_counts[id] > 1:
                    suffix = ''
                    if id and id[-1].isdigit():
                        suffix = '_'
                    new_id = f'{id}{suffix}{id_suffixes[id]}'
                    while new_id in set(resolved_ids):
                        id_suffixes[id] += 1
                        new_id = f'{id}{suffix}{id_suffixes[id]}'
                    resolved_ids[index] = new_id
                    id_suffixes[id] += 1
        assert len(resolved_ids) == len(set(resolved_ids)), f'Internal error: resolved_ids={resolved_ids!r}'
        return resolved_ids

    def _resolve_ids(self) -> Iterable[str]:
        """Resolve IDs for all ParameterSets (may contain duplicates)."""
        for idx, parameterset in enumerate(self.parametersets):
            if parameterset.id is not None:
                yield parameterset.id
            elif self.ids and idx < len(self.ids) and (self.ids[idx] is not None):
                yield self._idval_from_value_required(self.ids[idx], idx)
            else:
                yield '-'.join((self._idval(val, argname, idx) for val, argname in zip(parameterset.values, self.argnames)))

    def _idval(self, val: object, argname: str, idx: int) -> str:
        """Make an ID for a parameter in a ParameterSet."""
        idval = self._idval_from_function(val, argname, idx)
        if idval is not None:
            return idval
        idval = self._idval_from_hook(val, argname)
        if idval is not None:
            return idval
        idval = self._idval_from_value(val)
        if idval is not None:
            return idval
        return self._idval_from_argname(argname, idx)

    def _idval_from_function(self, val: object, argname: str, idx: int) -> Optional[str]:
        """Try to make an ID for a parameter in a ParameterSet using the
        user-provided id callable, if given."""
        if self.idfn is None:
            return None
        try:
            id = self.idfn(val)
        except Exception as e:
            prefix = f'{self.nodeid}: ' if self.nodeid is not None else ''
            msg = "error raised while trying to determine id of parameter '{}' at position {}"
            msg = prefix + msg.format(argname, idx)
            raise ValueError(msg) from e
        if id is None:
            return None
        return self._idval_from_value(id)

    def _idval_from_hook(self, val: object, argname: str) -> Optional[str]:
        """Try to make an ID for a parameter in a ParameterSet by calling the
        :hook:`pytest_make_parametrize_id` hook."""
        if self.config:
            id: Optional[str] = self.config.hook.pytest_make_parametrize_id(config=self.config, val=val, argname=argname)
            return id
        return None

    def _idval_from_value(self, val: object) -> Optional[str]:
        """Try to make an ID for a parameter in a ParameterSet from its value,
        if the value type is supported."""
        if isinstance(val, (str, bytes)):
            return _ascii_escaped_by_config(val, self.config)
        elif val is None or isinstance(val, (float, int, bool, complex)):
            return str(val)
        elif isinstance(val, Pattern):
            return ascii_escaped(val.pattern)
        elif val is NOTSET:
            pass
        elif isinstance(val, enum.Enum):
            return str(val)
        elif isinstance(getattr(val, '__name__', None), str):
            name: str = getattr(val, '__name__')
            return name
        return None

    def _idval_from_value_required(self, val: object, idx: int) -> str:
        """Like _idval_from_value(), but fails if the type is not supported."""
        id = self._idval_from_value(val)
        if id is not None:
            return id
        if self.func_name is not None:
            prefix = f'In {self.func_name}: '
        elif self.nodeid is not None:
            prefix = f'In {self.nodeid}: '
        else:
            prefix = ''
        msg = f'{prefix}ids contains unsupported value {saferepr(val)} (type: {type(val)!r}) at index {idx}. Supported types are: str, bytes, int, float, complex, bool, enum, regex or anything with a __name__.'
        fail(msg, pytrace=False)

    @staticmethod
    def _idval_from_argname(argname: str, idx: int) -> str:
        """Make an ID for a parameter in a ParameterSet from the argument name
        and the index of the ParameterSet."""
        return str(argname) + str(idx)