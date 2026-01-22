from abc import ABC, abstractmethod
import copy
from threading import Lock
from typing import Dict, Iterable, List, Optional
from .metrics_core import Metric
class _EmptyCollector(Collector):

    def collect(self) -> Iterable[Metric]:
        return []