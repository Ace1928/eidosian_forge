from enum import IntEnum
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
class IntMap(dict):

    def __init__(self):
        dict.__init__(self)
        self.added_keys = []

    def emplace(self, key, value):
        if not isinstance(key, (int, str)):
            raise TypeError(f'key must be a int or str not {type(key)}.')
        if not isinstance(value, NgramPart):
            raise TypeError(f'value must be a NGramPart not {type(value)}.')
        if key not in self:
            self.added_keys.append(key)
            self[key] = value
        return self[key]

    def __repr__(self):
        vals = {k: repr(v) for k, v in self.items()}
        rows = ['{']
        for k, v in sorted(vals.items()):
            if '\n' in v:
                vs = v.split('\n')
                for i, line in enumerate(vs):
                    if i == 0:
                        if line == '{':
                            rows.append(f'  {k}={line}')
                        else:
                            rows.append(f'  {k}={line},')
                    elif i == len(vs) - 1:
                        rows.append(f'  {line}')
                    else:
                        rows.append(f'    {line}')
            else:
                rows.append(f'  {k}={v},')
        rows.append('}')
        return '\n'.join(rows)

    @property
    def first_key(self):
        if len(self) == 0:
            raise ValueError('IntMap is empty.')
        return self.added_keys[0]