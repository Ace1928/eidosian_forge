import ast
import asyncio
import inspect
from functools import wraps

        We need the dummy no-op async def to protect from
        trio's internal. See https://github.com/python-trio/trio/issues/89
        