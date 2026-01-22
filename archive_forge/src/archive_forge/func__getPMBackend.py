from reportlab.graphics.shapes import *
from reportlab.graphics.renderbase import getStateDelta, renderScaledDrawing
from reportlab.pdfbase.pdfmetrics import getFont, unicode2T1
from reportlab.lib.utils import isUnicode
from reportlab import rl_config
from .utils import setFont as _setFont, RenderPMError
import os, sys
from io import BytesIO, StringIO
from math import sin, cos, pi, ceil
from reportlab.graphics.renderbase import Renderer
def _getPMBackend(backend=None):
    if not backend:
        backend = rl_config.renderPMBackend
    if backend == '_renderPM':
        try:
            import _rl_renderPM as M
        except ImportError as errMsg:
            try:
                import rlPyCairo as M
            except ImportError:
                raise RenderPMError('Cannot import desired renderPM backend, {backend}.\nNo module named _rl_renderPM\nit may be badly or not installed!\nYou may need to install development tools\nor seek advice at the users list see\nhttps://pairlist2.pair.net/mailman/listinfo/reportlab-users')
    elif 'cairo' in backend.lower():
        try:
            import rlPyCairo as M
        except ImportError as errMsg:
            try:
                import _rl_renderPM as M
            except ImportError:
                raise RenderPMError(f'cannot import desired renderPM backend {backend}\nSeek advice at the users list see\nhttps://pairlist2.pair.net/mailman/listinfo/reportlab-users')
    else:
        raise RenderPMError(f'Invalid renderPM backend, {backend}')
    return M