import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def _refresh_labels(self):
    lbls = self.labels()
    self._labels = [l[0] for _, l in enumerate(lbls)]