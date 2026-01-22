from __future__ import annotations
from typing import Final
import streamlit as st
from streamlit import config
from streamlit.errors import UncaughtAppException
from streamlit.logger import get_logger
def _print_rich_exception(e: BaseException) -> None:
    from rich import box, panel

    class ConfigurablePanel(panel.Panel):

        def __init__(self, renderable, box=box.Box('────\n    \n────\n    \n────\n────\n    \n────\n'), **kwargs):
            super().__init__(renderable, box, **kwargs)
    from rich import traceback as rich_traceback
    rich_traceback.Panel = ConfigurablePanel
    from rich.console import Console
    console = Console(color_system='256', force_terminal=True, width=88, no_color=False, tab_size=8)
    import streamlit.runtime.scriptrunner.script_runner as script_runner
    console.print(rich_traceback.Traceback.from_exception(type(e), e, e.__traceback__, width=88, show_locals=False, max_frames=100, word_wrap=False, extra_lines=3, suppress=[script_runner]))