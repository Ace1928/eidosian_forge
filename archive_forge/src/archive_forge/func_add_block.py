from typing import Any, List
import ray
from ray import cloudpickle
def add_block(self, block: List[Any]) -> None:
    if self._count < 10:
        for i in range(min(10 - self._count, len(block))):
            self._running_mean.add(self._real_size(block[i]), weight=1)
    if self._count < 100:
        for i in range(10 - self._count % 10, min(100 - self._count, len(block)), 10):
            self._running_mean.add(self._real_size(block[i]), weight=10)
    if (len(block) + self._count % 100) // 100 > 1:
        for i in range(100 - self._count % 100, len(block), 100):
            self._running_mean.add(self._real_size(block[i]), weight=100)
    self._count += len(block)