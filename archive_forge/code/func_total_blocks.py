from __future__ import annotations
import os
from math import ceil
from kombu.utils.objects import cached_property
@property
def total_blocks(self) -> float:
    return self.stat.f_blocks * self.stat.f_frsize / 1024