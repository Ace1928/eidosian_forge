from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Any
from ..models.ui import UIElement
from ..util.browser import NEW_PARAM, get_browser_controller
from .notebook import DEFAULT_JUPYTER_URL, ProxyUrlFunc, run_notebook_hook
from .saving import save
from .state import curstate
def _show_with_state(obj: UIElement, state: State, browser: str | None, new: BrowserTarget, notebook_handle: bool=False) -> CommsHandle | None:
    """

    """
    controller = get_browser_controller(browser=browser)
    comms_handle = None
    shown = False
    if state.notebook:
        assert state.notebook_type is not None
        comms_handle = run_notebook_hook(state.notebook_type, 'doc', obj, state, notebook_handle)
        shown = True
    if state.file or not shown:
        _show_file_with_state(obj, state, new, controller)
    return comms_handle