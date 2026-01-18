from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
Indicates whether the transformation can be applied by a
        subprocessing pool. This should be overridden to return True for
        transformations that the transmuter can parallelize.
        