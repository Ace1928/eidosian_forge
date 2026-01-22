from __future__ import annotations
import logging
import re
import sys
import typing as t
from datetime import datetime
from datetime import timezone
class _Missing:

    def __repr__(self) -> str:
        return 'no value'

    def __reduce__(self) -> str:
        return '_missing'