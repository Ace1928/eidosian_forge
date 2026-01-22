from __future__ import annotations
import datetime
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.layout.dimension import AnyDimension, D
from prompt_toolkit.layout.utils import explode_text_fragments
from prompt_toolkit.utils import get_cwidth
def _add_suffix(self, label: AnyFormattedText) -> StyleAndTextTuples:
    label = to_formatted_text(label, style='class:label')
    return label + [('', self.suffix)]