from dataclasses import dataclass, field
from collections import defaultdict
import copy
import torch
from torch.fx import (
from torch.fx._compatibility import compatibility
from typing import Dict, List, Set, Any, Union, Tuple
import logging
import os
def _remove_overlapping_matches(self, matches: List[InternalMatch]) -> List[InternalMatch]:
    non_overlapping_matches: List[InternalMatch] = list()
    nodes_matched: Set[Node] = set()
    for match in matches:
        found_overlap = False
        for pn, gn in match.nodes_map.items():
            if pn.op not in {'placeholder', 'output'} and gn in nodes_matched:
                found_overlap = True
                break
        if not found_overlap:
            non_overlapping_matches.append(match)
            for pn, gn in match.nodes_map.items():
                if pn.op not in {'placeholder', 'output'}:
                    nodes_matched.add(gn)
    return non_overlapping_matches