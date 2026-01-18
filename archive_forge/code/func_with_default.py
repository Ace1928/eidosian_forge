import asyncio
import functools
from typing import Tuple
def with_default(self, *, x='x'):
    print('x: ' + x)