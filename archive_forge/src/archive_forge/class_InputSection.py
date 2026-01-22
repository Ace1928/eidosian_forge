import os
import os.path
from warnings import warn
from subprocess import Popen, PIPE
import numpy as np
import ase.io
from ase.units import Rydberg
from ase.calculators.calculator import (Calculator, all_changes, Parameters,
class InputSection:
    """Represents a section of a CP2K input file"""

    def __init__(self, name, params=None):
        self.name = name.upper()
        self.params = params
        self.keywords = []
        self.subsections = []

    def write(self):
        """Outputs input section as string"""
        output = []
        for k in self.keywords:
            output.append(k)
        for s in self.subsections:
            if s.params:
                output.append('&%s %s' % (s.name, s.params))
            else:
                output.append('&%s' % s.name)
            for l in s.write():
                output.append('   %s' % l)
            output.append('&END %s' % s.name)
        return output

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

    def get_subsection(self, path):
        """Finds a subsection"""
        parts = path.upper().split('/', 1)
        candidates = [s for s in self.subsections if s.name == parts[0]]
        if len(candidates) > 1:
            raise Exception('Multiple %s sections found ' % parts[0])
        if len(candidates) == 0:
            s = InputSection(name=parts[0])
            self.subsections.append(s)
            candidates = [s]
        if len(parts) == 1:
            return candidates[0]
        return candidates[0].get_subsection(parts[1])