import sys
import re
from abc import ABCMeta
from unicodedata import name as unicode_name
from decimal import Decimal, DecimalException
from typing import Any, cast, overload, Callable, Dict, Generic, List, \
class ParserMeta(ABCMeta):
    token_base_class: Type[Any]
    literals_pattern: Pattern[str]
    name_pattern: Pattern[str]
    tokenizer: Optional[Pattern[str]]
    symbol_table: MutableMapping[str, Type[Any]]

    def __new__(mcs, name: str, bases: Tuple[Type[Any], ...], namespace: Dict[str, Any]) -> 'ParserMeta':
        cls = super(ParserMeta, mcs).__new__(mcs, name, bases, namespace)
        for k, v in sys.modules[cls.__module__].__dict__.items():
            if isinstance(v, ParserMeta) and v.__module__ == cls.__module__:
                raise RuntimeError('Multiple parser class definitions per module are not allowed')
        if not hasattr(cls, 'token_base_class'):
            cls.token_base_class = Token
        if not hasattr(cls, 'literals_pattern'):
            cls.literals_pattern = re.compile('\'[^\']*\'|"[^"]*"|(?:\\d+|\\.\\d+)(?:\\.\\d*)?(?:[Ee][+-]?\\d+)?')
        if not hasattr(cls, 'name_pattern'):
            cls.name_pattern = re.compile('[A-Za-z0-9_]+')
        if 'tokenizer' not in namespace:
            cls.tokenizer = None
        if 'symbol_table' not in namespace:
            cls.symbol_table = {}
            for base_class in bases:
                if hasattr(base_class, 'symbol_table'):
                    cls.symbol_table.update(base_class.symbol_table)
                    break
        return cls