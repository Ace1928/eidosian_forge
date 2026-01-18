from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def wait_until_render_complete(driver: WebDriver, timeout: int) -> None:
    """

    """
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.support.wait import WebDriverWait

    def is_bokeh_loaded(driver: WebDriver) -> bool:
        return cast(bool, driver.execute_script('\n            return typeof Bokeh !== "undefined" && Bokeh.documents != null && Bokeh.documents.length != 0\n        '))
    try:
        WebDriverWait(driver, timeout, poll_frequency=0.1).until(is_bokeh_loaded)
    except TimeoutException as e:
        _log_console(driver)
        raise RuntimeError('Bokeh was not loaded in time. Something may have gone wrong.') from e
    driver.execute_script(_WAIT_SCRIPT)

    def is_bokeh_render_complete(driver: WebDriver) -> bool:
        return cast(bool, driver.execute_script('return window._bokeh_render_complete;'))
    try:
        WebDriverWait(driver, timeout, poll_frequency=0.1).until(is_bokeh_render_complete)
    except TimeoutException:
        log.warning("The webdriver raised a TimeoutException while waiting for a 'bokeh:idle' event to signify that the layout has rendered. Something may have gone wrong.")
    finally:
        _log_console(driver)