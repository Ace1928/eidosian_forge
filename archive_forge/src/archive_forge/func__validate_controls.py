import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
@validate('controls')
def _validate_controls(self, proposal):
    """Validate controls list.

        Makes sure only one instance of any given layer can exist in the
        controls list.
        """
    self._control_ids = [c.model_id for c in proposal.value]
    if len(set(self._control_ids)) != len(self._control_ids):
        raise ControlException('duplicate control detected, only use each control once')
    return proposal.value