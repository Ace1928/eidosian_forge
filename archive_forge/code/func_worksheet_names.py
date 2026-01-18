from __future__ import annotations
import os
import abc
from lazyops.libs.pooler import ThreadPooler
from typing import Optional, List, Dict, Any, Union, Type, Tuple, Generator, TYPE_CHECKING
from lazyops.imports._gspread import resolve_gspread
import gspread
@property
def worksheet_names(self) -> List[str]:
    """
        Return the list of worksheet names
        """
    if self._worksheet_names is None:
        self._worksheet_names = [sheet.title for sheet in self.sheet.worksheets(False)]
    return self._worksheet_names