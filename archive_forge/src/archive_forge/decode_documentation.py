from .checks import check_data
from .specs import (
Decode message bytes and return messages as a dictionary.

    Raises ValueError if the bytes are out of range or the message is
    invalid.

    This is not a part of the public API.
    