import asyncio
import functools
from typing import Tuple
class HelpTextComponent:

    def identity(self, *, alpha, beta='0'):
        return (alpha, beta)