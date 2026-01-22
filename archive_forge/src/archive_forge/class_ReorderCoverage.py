from fontTools import ttLib
from fontTools.ttLib.tables import otBase
from fontTools.ttLib.tables import otTables as ot
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque
from typing import (
@dataclass(frozen=True)
class ReorderCoverage(ReorderRule):
    """Reorder a Coverage table, and optionally a list that is sorted parallel to it."""
    parallel_list_attr: Optional[str] = None
    coverage_attr: str = _COVERAGE_ATTR

    def apply(self, font: ttLib.TTFont, value: otBase.BaseTable) -> None:
        coverage = _get_dotted_attr(value, self.coverage_attr)
        if type(coverage) is not list:
            parallel_list = None
            if self.parallel_list_attr:
                parallel_list = _get_dotted_attr(value, self.parallel_list_attr)
                assert type(parallel_list) is list, f'{self.parallel_list_attr} should be a list'
                assert len(parallel_list) == len(coverage.glyphs), 'Nothing makes sense'
            _sort_by_gid(font.getGlyphID, coverage.glyphs, parallel_list)
        else:
            assert not self.parallel_list_attr, f"Can't have multiple coverage AND a parallel list; {self}"
            for coverage_entry in coverage:
                _sort_by_gid(font.getGlyphID, coverage_entry.glyphs, None)