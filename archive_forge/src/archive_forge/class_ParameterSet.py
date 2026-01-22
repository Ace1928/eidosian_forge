import collections.abc
import dataclasses
import inspect
from typing import Any
from typing import Callable
from typing import Collection
from typing import final
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from .._code import getfslineno
from ..compat import ascii_escaped
from ..compat import NOTSET
from ..compat import NotSetType
from _pytest.config import Config
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.outcomes import fail
from _pytest.warning_types import PytestUnknownMarkWarning
class ParameterSet(NamedTuple):
    values: Sequence[Union[object, NotSetType]]
    marks: Collection[Union['MarkDecorator', 'Mark']]
    id: Optional[str]

    @classmethod
    def param(cls, *values: object, marks: Union['MarkDecorator', Collection[Union['MarkDecorator', 'Mark']]]=(), id: Optional[str]=None) -> 'ParameterSet':
        if isinstance(marks, MarkDecorator):
            marks = (marks,)
        else:
            assert isinstance(marks, collections.abc.Collection)
        if id is not None:
            if not isinstance(id, str):
                raise TypeError(f'Expected id to be a string, got {type(id)}: {id!r}')
            id = ascii_escaped(id)
        return cls(values, marks, id)

    @classmethod
    def extract_from(cls, parameterset: Union['ParameterSet', Sequence[object], object], force_tuple: bool=False) -> 'ParameterSet':
        """Extract from an object or objects.

        :param parameterset:
            A legacy style parameterset that may or may not be a tuple,
            and may or may not be wrapped into a mess of mark objects.

        :param force_tuple:
            Enforce tuple wrapping so single argument tuple values
            don't get decomposed and break tests.
        """
        if isinstance(parameterset, cls):
            return parameterset
        if force_tuple:
            return cls.param(parameterset)
        else:
            return cls(parameterset, marks=[], id=None)

    @staticmethod
    def _parse_parametrize_args(argnames: Union[str, Sequence[str]], argvalues: Iterable[Union['ParameterSet', Sequence[object], object]], *args, **kwargs) -> Tuple[Sequence[str], bool]:
        if isinstance(argnames, str):
            argnames = [x.strip() for x in argnames.split(',') if x.strip()]
            force_tuple = len(argnames) == 1
        else:
            force_tuple = False
        return (argnames, force_tuple)

    @staticmethod
    def _parse_parametrize_parameters(argvalues: Iterable[Union['ParameterSet', Sequence[object], object]], force_tuple: bool) -> List['ParameterSet']:
        return [ParameterSet.extract_from(x, force_tuple=force_tuple) for x in argvalues]

    @classmethod
    def _for_parametrize(cls, argnames: Union[str, Sequence[str]], argvalues: Iterable[Union['ParameterSet', Sequence[object], object]], func, config: Config, nodeid: str) -> Tuple[Sequence[str], List['ParameterSet']]:
        argnames, force_tuple = cls._parse_parametrize_args(argnames, argvalues)
        parameters = cls._parse_parametrize_parameters(argvalues, force_tuple)
        del argvalues
        if parameters:
            for param in parameters:
                if len(param.values) != len(argnames):
                    msg = '{nodeid}: in "parametrize" the number of names ({names_len}):\n  {names}\nmust be equal to the number of values ({values_len}):\n  {values}'
                    fail(msg.format(nodeid=nodeid, values=param.values, names=argnames, names_len=len(argnames), values_len=len(param.values)), pytrace=False)
        else:
            mark = get_empty_parameterset_mark(config, argnames, func)
            parameters.append(ParameterSet(values=(NOTSET,) * len(argnames), marks=[mark], id=None))
        return (argnames, parameters)