import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
class XPathArray(XPathFunction):
    """
    A token for processing XPath 3.1+ arrays.
    """
    symbol = 'array'
    label = 'array'
    pattern = '(?<!\\$)\\barray(?=\\s*(?:\\(\\:.*\\:\\))?\\s*\\{(?!\\:))'
    _array: Optional[List[Any]] = None

    def __init__(self, parser: XPathParserType, items: Optional[Iterable[Any]]=None) -> None:
        if items is not None:
            self._array = [x for x in items]
        super().__init__(parser)

    def __repr__(self) -> str:
        if self._array is not None:
            return f'<{self.__class__.__name__} object at {hex(id(self))}>'
        return '<{} object (not evaluated constructor) at {}>'.format(self.__class__.__name__, hex(id(self)))

    def __str__(self) -> str:
        if self._array is not None:
            return str(self._array)
        items_desc = f'{len(self)} items' if len(self) != 1 else '1 item'
        if self.symbol == 'array':
            return f'not evaluated curly array constructor with {items_desc}'
        return f'not evaluated square array constructor with {items_desc}'

    def __len__(self) -> int:
        if self._array is None:
            return len(self._items)
        return len(self._array)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, XPathArray):
            if self._array is None or other._array is None:
                raise ElementPathValueError('cannot compare not evaluated arrays')
            return self._array == other._array
        return NotImplemented

    @property
    def source(self) -> str:
        if self._array is None:
            items = ', '.join((f'{tk.source}' for tk in self))
        else:
            items = ', '.join((f'{v!r}' for v in self._array))
        return f'array{{{items}}}' if self.symbol == 'array' else f'[{items}]'

    def nud(self) -> 'XPathArray':
        self.value = None
        self.parser.advance('{')
        del self._items[:]
        if self.parser.next_token.symbol not in ('}', '(end)'):
            while True:
                self._items.append(self.parser.expression(5))
                if self.parser.next_token.symbol != ',':
                    break
                self.parser.advance()
        self.parser.advance('}')
        return self

    def evaluate(self, context: ContextArgType=None) -> 'XPathArray':
        if self._array is not None:
            return self
        return XPathArray(self.parser, items=self._evaluate(context))

    def _evaluate(self, context: ContextArgType=None) -> List[Any]:
        if self.symbol == 'array':
            items: List[Any] = []
            for tk in self._items:
                items.extend(tk.select(context))
            return items
        else:
            return [tk.evaluate(context) for tk in self._items]

    def __call__(self, *args: XPathFunctionArgType, context: ContextArgType=None) -> Any:
        if len(args) != 1 or not isinstance(args[0], int):
            raise self.error('XPTY0004', 'exactly one xs:integer argument is expected')
        position = args[0]
        if position <= 0:
            raise self.error('FOAY0001')
        if self._array is not None:
            items = self._array
        else:
            items = self._evaluate(context)
        try:
            return items[position - 1]
        except IndexError:
            raise self.error('FOAY0001')

    def items(self, context: ContextArgType=None) -> List[Any]:
        if self._array is not None:
            return self._array.copy()
        return self._evaluate(context)

    def iter_flatten(self, context: ContextArgType=None) -> Iterator[Any]:
        if self._array is not None:
            items = self._array
        else:
            items = self._evaluate(context)
        for item in items:
            if isinstance(item, XPathArray):
                yield from item.iter_flatten(context)
            elif isinstance(item, list):
                yield from item
            else:
                yield item

    def match_function_test(self, function_test: Union[str, List[Any]], as_argument: bool=False) -> bool:
        if isinstance(function_test, list):
            sequence_types = function_test
        else:
            sequence_types = split_function_test(function_test)
        if not sequence_types or not sequence_types[-1]:
            return False
        elif sequence_types[0] == '*':
            return True
        elif len(sequence_types) != 2:
            return False
        index_type, value_type = sequence_types
        if index_type.endswith(('+', '*')):
            return False
        return match_sequence_type(1, index_type) and all((match_sequence_type(v, value_type, self.parser) for v in self.items()))