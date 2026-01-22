import asyncio
import contextvars
import unittest
from test import support
class LacksExit:

    async def __aenter__(self):
        pass