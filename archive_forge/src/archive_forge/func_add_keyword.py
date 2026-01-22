import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import (Calculator, all_changes, Parameters,
def add_keyword(self, path, line, unique=True):
    """Adds a keyword to section."""
    parts = path.upper().split('/', 1)
    candidates = [s for s in self.subsections if s.name == parts[0]]
    if len(candidates) == 0:
        s = InputSection(name=parts[0])
        self.subsections.append(s)
        candidates = [s]
    elif len(candidates) != 1:
        raise Exception('Multiple %s sections found ' % parts[0])
    key = line.split()[0].upper()
    if len(parts) > 1:
        candidates[0].add_keyword(parts[1], line, unique)
    elif key == '_SECTION_PARAMETERS_':
        if candidates[0].params is not None:
            msg = 'Section parameter of section %s already set' % parts[0]
            raise Exception(msg)
        candidates[0].params = line.split(' ', 1)[1].strip()
    else:
        old_keys = [k.split()[0].upper() for k in candidates[0].keywords]
        if unique and key in old_keys:
            msg = 'Keyword %s already present in section %s'
            raise Exception(msg % (key, parts[0]))
        candidates[0].keywords.append(line)