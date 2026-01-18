from __future__ import annotations
import hashlib
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, cast
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import remove_missing
from ..exceptions import PlotnineError, PlotnineWarning
from ..mapping.aes import rename_aesthetics
from .guide import GuideElements, guide

        Heights of the keys

        If legend is horizontal, then key heights must be equal, so we
        use the maximum
        