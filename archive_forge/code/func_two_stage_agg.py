from __future__ import annotations
from enum import Enum
from typing import NamedTuple, TYPE_CHECKING
def two_stage_agg(antialias_stage_2: UnzippedAntialiasStage2 | None):
    """Information used to perform the correct stage 2 aggregation."""
    if not antialias_stage_2:
        return (False, False)
    aa_combinations = antialias_stage_2[0]
    use_2_stage_agg = False
    for comb in aa_combinations:
        if comb in (AntialiasCombination.SUM_2AGG, AntialiasCombination.MIN, AntialiasCombination.FIRST, AntialiasCombination.LAST):
            use_2_stage_agg = True
            break
    overwrite = True
    for comb in aa_combinations:
        if comb == AntialiasCombination.SUM_1AGG:
            overwrite = False
            break
    return (overwrite, use_2_stage_agg)