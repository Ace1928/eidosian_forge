from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class ListUniqueItemsError(PydanticValueError):
    code = 'list.unique_items'
    msg_template = 'the list has duplicated items'