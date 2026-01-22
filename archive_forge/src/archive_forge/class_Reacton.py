from __future__ import annotations
import os
from typing import (
import param
from param.parameterized import register_reference_transform
from pyviz_comms import JupyterComm
from ..config import config
from ..models import IPyWidget as _BkIPyWidget
from .base import PaneBase
class Reacton(IPyWidget):

    def __init__(self, object=None, **params):
        super().__init__(object=object, **params)
        self._rcs = {}

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        return getattr(obj, '__module__', 'None').startswith('reacton')

    def _cleanup(self, root: Model | None=None) -> None:
        if root and root.ref['id'] in self._rcs:
            rc = self._rcs.pop(root.ref['id'])
            try:
                rc.close()
            except Exception:
                pass
        super()._cleanup(root)

    def _get_ipywidget(self, obj, doc: Document, root: Model, comm: Optional[Comm], **kwargs):
        if not isinstance(comm, JupyterComm) or 'PANEL_IPYWIDGET' in os.environ:
            from ..io.ipywidget import Widget
        import reacton
        widget, rc = reacton.render(obj)
        self._rcs[root.ref['id']] = rc
        return super()._get_ipywidget(widget, doc, root, comm, **kwargs)