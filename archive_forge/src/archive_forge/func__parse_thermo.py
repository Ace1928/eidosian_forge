from __future__ import annotations
import re
from glob import glob
from io import StringIO
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
from monty.io import zopen
from monty.json import MSONable
from pymatgen.io.lammps.data import LammpsBox
def _parse_thermo(lines: list[str]) -> pd.DataFrame:
    multi_pattern = '-+\\s+Step\\s+([0-9]+)\\s+-+'
    if re.match(multi_pattern, lines[0]):
        timestep_marks = [idx for idx, line in enumerate(lines) if re.match(multi_pattern, line)]
        timesteps = np.split(lines, timestep_marks)[1:]
        dicts = []
        kv_pattern = '([0-9A-Za-z_\\[\\]]+)\\s+=\\s+([0-9eE\\.+-]+)'
        for ts in timesteps:
            data = {}
            step = re.match(multi_pattern, ts[0])
            assert step is not None
            data['Step'] = int(step[1])
            data.update({k: float(v) for k, v in re.findall(kv_pattern, ''.join(ts[1:]))})
            dicts.append(data)
        df = pd.DataFrame(dicts)
        columns = ['Step'] + [k for k, v in re.findall(kv_pattern, ''.join(timesteps[0][1:]))]
        df = df[columns]
    else:
        df = pd.read_csv(StringIO(''.join(lines)), delim_whitespace=True)
    return df