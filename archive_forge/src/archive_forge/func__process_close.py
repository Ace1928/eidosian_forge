from __future__ import annotations
from collections import defaultdict
from typing import (
import param
from bokeh.models import Spacer as BkSpacer, TabPanel as BkTabPanel
from ..models.tabs import Tabs as BkTabs
from ..viewable import Layoutable
from .base import NamedListPanel
def _process_close(self, ref, attr, old, new):
    """
        Handle closed tabs.
        """
    model, _ = self._models.get(ref)
    if model:
        try:
            inds = [old.index(tab) for tab in new]
        except Exception:
            return (old, None)
        old = self.objects
        new = [old[i] for i in inds]
    return (old, new)