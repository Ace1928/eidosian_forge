from typing import Any
import numpy as np
class SVMAttributes:

    def __init__(self):
        self._names = []

    def add(self, name: str, value: Any) -> None:
        if isinstance(value, list) and name not in {'kernel_params'}:
            if name in {'vectors_per_class'}:
                value = np.array(value, dtype=np.int64)
            else:
                value = np.array(value, dtype=np.float32)
        setattr(self, name, value)

    def __str__(self) -> str:
        rows = ['Attributes']
        for name in self._names:
            rows.append(f'  {name}={getattr(self, name)}')
        return '\n'.join(rows)