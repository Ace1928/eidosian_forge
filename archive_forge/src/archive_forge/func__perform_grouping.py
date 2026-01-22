from __future__ import annotations
import collections
import csv
import datetime
import itertools
import json
import logging
import multiprocessing as mp
import re
from typing import TYPE_CHECKING, Literal
from monty.json import MontyDecoder, MontyEncoder, MSONable
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.analysis.structure_matcher import SpeciesComparator, StructureMatcher
from pymatgen.core import Composition, Element
def _perform_grouping(args):
    entries_json, hosts_json, ltol, stol, angle_tol, primitive_cell, scale, comparator, groups = args
    entries = json.loads(entries_json, cls=MontyDecoder)
    hosts = json.loads(hosts_json, cls=MontyDecoder)
    unmatched = list(zip(entries, hosts))
    while len(unmatched) > 0:
        ref_host = unmatched[0][1]
        logger.info(f'Reference tid = {unmatched[0][0].entry_id}, formula = {ref_host.formula}')
        ref_formula = ref_host.reduced_formula
        logger.info(f'Reference host = {ref_formula}')
        matches = [unmatched[0]]
        for idx in range(1, len(unmatched)):
            test_host = unmatched[idx][1]
            logger.info(f'Testing tid = {unmatched[idx][0].entry_id}, formula = {test_host.formula}')
            test_formula = test_host.reduced_formula
            logger.info(f'Test host = {test_formula}')
            matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, primitive_cell=primitive_cell, scale=scale, comparator=comparator)
            if matcher.fit(ref_host, test_host):
                logger.info('Fit found')
                matches.append(unmatched[idx])
        groups.append(json.dumps([m[0] for m in matches], cls=MontyEncoder))
        unmatched = list(filter(lambda x: x not in matches, unmatched))
        logger.info(f'{len(unmatched)} unmatched remaining')