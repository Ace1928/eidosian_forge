from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Set, Tuple, Type, Union
from .typing import display_as_type
class NotDigitError(PydanticValueError):
    code = 'payment_card_number.digits'
    msg_template = 'card number is not all digits'