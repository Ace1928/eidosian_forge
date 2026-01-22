from typing import Awaitable

ValidationRule = Callable[[Any], Awaitable[bool]]


async def is_valid_array(value) -> Awaitable[bool]:
    return isinstance(value, list)
