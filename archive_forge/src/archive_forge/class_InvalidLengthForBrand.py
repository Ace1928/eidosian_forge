from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class InvalidLengthForBrand(PydanticValueError):
    code = 'payment_card_number.invalid_length_for_brand'
    msg_template = 'Length for a {brand} card must be {required_length}'