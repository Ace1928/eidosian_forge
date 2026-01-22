from __future__ import annotations
from collections import defaultdict
from typing import (
import param
from bokeh.models import Spacer as BkSpacer, TabPanel as BkTabPanel
from ..models.tabs import Tabs as BkTabs
from ..viewable import Layoutable
from .base import NamedListPanel
@property
def _preprocess_params(self):
    return NamedListPanel._preprocess_params + (['active'] if self.dynamic else [])