import operator
import os
import re
import subprocess
import tempfile
from functools import reduce
from optparse import OptionParser
from nltk.internals import find_binary
from nltk.sem.drt import (
from nltk.sem.logic import (
def _parse_to_drs_dict(self, boxer_out, use_disc_id):
    lines = boxer_out.decode('utf-8').split('\n')
    drs_dict = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('id('):
            comma_idx = line.index(',')
            discourse_id = line[3:comma_idx]
            if discourse_id[0] == "'" and discourse_id[-1] == "'":
                discourse_id = discourse_id[1:-1]
            drs_id = line[comma_idx + 1:line.index(')')]
            i += 1
            line = lines[i]
            assert line.startswith(f'sem({drs_id},')
            if line[-4:] == "').'":
                line = line[:-4] + ').'
            assert line.endswith(').'), f"can't parse line: {line}"
            search_start = len(f'sem({drs_id},[')
            brace_count = 1
            drs_start = -1
            for j, c in enumerate(line[search_start:]):
                if c == '[':
                    brace_count += 1
                if c == ']':
                    brace_count -= 1
                    if brace_count == 0:
                        drs_start = search_start + j + 1
                        if line[drs_start:drs_start + 3] == "','":
                            drs_start = drs_start + 3
                        else:
                            drs_start = drs_start + 1
                        break
            assert drs_start > -1
            drs_input = line[drs_start:-2].strip()
            parsed = self._parse_drs(drs_input, discourse_id, use_disc_id)
            drs_dict[discourse_id] = self._boxer_drs_interpreter.interpret(parsed)
        i += 1
    return drs_dict