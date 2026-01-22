import json
import random
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from typing import List as LList
from pydantic import BaseModel, ConfigDict, Field, validator
from pydantic.alias_generators import to_camel
def _generate_name(length: int=12) -> str:
    """Generate a random name.

    This implementation roughly based the following snippet in core:
    https://github.com/wandb/core/blob/master/lib/js/cg/src/utils/string.ts#L39-L44.
    """

    def base_repr(number: int, base: int, padding: int=0) -> str:
        digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        if base > len(digits):
            raise ValueError('Bases greater than 36 not handled in base_repr.')
        elif base < 2:
            raise ValueError('Bases less than 2 not handled in base_repr.')
        num = abs(number)
        res = []
        while num:
            res.append(digits[num % base])
            num //= base
        if padding:
            res.append('0' * padding)
        if number < 0:
            res.append('-')
        return ''.join(reversed(res or '0'))
    rand = random.random()
    rand = int(float(str(rand)[2:]))
    rand36 = base_repr(rand, 36)
    return rand36.lower()[:length]