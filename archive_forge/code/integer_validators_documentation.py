import typing
from typing import Awaitable, Callable, Any, List, Union
import asyncio

        Validates if the provided integer value is positive and not contained within a specified list of integers.

        Args:
            value (Any): The value to validate.
            values (List[int]): The list of integers to check against.

        Returns:
            bool: True if the value is positive and not contained within the list, False otherwise.
        