from __future__ import annotations
from collections import defaultdict
from typing import (
import param
from bokeh.models import Spacer as BkSpacer, TabPanel as BkTabPanel
from ..models.tabs import Tabs as BkTabs
from ..viewable import Layoutable
from .base import NamedListPanel

        Returns new child models for the layout while reusing unchanged
        models and cleaning up any dropped objects.
        