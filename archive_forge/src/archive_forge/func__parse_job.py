from __future__ import annotations
import os
import re
import shutil
import subprocess
from string import Template
from typing import TYPE_CHECKING
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core.structure import Molecule
@staticmethod
def _parse_job(output):
    BSE_exitons_patt = re.compile('^exiton \\s+ (\\d+)  : \\s+  ([\\d.]+) \\( \\s+ ([-\\d.]+) \\) \\s+ \\| .*  ', re.VERBOSE)
    end_patt = re.compile('\\s*program returned normally\\s*')
    total_time_patt = re.compile('\\s*total \\s+ time: \\s+  ([\\d.]+) .*', re.VERBOSE)
    BSE_results = {}
    parse_BSE_results = False
    parse_total_time = False
    for line in output.split('\n'):
        if parse_total_time:
            m = end_patt.search(line)
            if m:
                BSE_results.update(end_normally=True)
            m = total_time_patt.search(line)
            if m:
                BSE_results.update(total_time=m.group(1))
        if parse_BSE_results:
            if line.find('FULL BSE main valence -> conduction transitions weight:') != -1:
                parse_total_time = True
                parse_BSE_results = False
                continue
            m = BSE_exitons_patt.search(line)
            if m:
                dct = {}
                dct.update(bse_eig=m.group(2), osc_strength=m.group(3))
                BSE_results[str(m.group(1).strip())] = dct
        if line.find('FULL BSE eig.(eV), osc. strength and dipoles:') != -1:
            parse_BSE_results = True
    return BSE_results