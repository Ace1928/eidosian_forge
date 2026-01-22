from contextlib import contextmanager
from copy import deepcopy
from enum import IntEnum
import re
def _sortAxisValues(axisValues):
    results = []
    seenAxes = set()
    format4 = sorted([v for v in axisValues if v.Format == 4], key=lambda v: len(v.AxisValueRecord), reverse=True)
    for val in format4:
        axisIndexes = set((r.AxisIndex for r in val.AxisValueRecord))
        minIndex = min(axisIndexes)
        if not seenAxes & axisIndexes:
            seenAxes |= axisIndexes
            results.append((minIndex, val))
    for val in axisValues:
        if val in format4:
            continue
        axisIndex = val.AxisIndex
        if axisIndex not in seenAxes:
            seenAxes.add(axisIndex)
            results.append((axisIndex, val))
    return [axisValue for _, axisValue in sorted(results)]