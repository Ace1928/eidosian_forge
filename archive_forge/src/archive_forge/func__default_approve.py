from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
def _default_approve(_input: str) -> bool:
    msg = "Do you approve of the following input? Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    msg += '\n\n' + _input + '\n'
    resp = input(msg)
    return resp.lower() in ('yes', 'y')