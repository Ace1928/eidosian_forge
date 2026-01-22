from __future__ import annotations
import logging # isort:skip
from ..util.dependencies import import_required # isort:skip
import_required("selenium.webdriver",
import atexit
import os
from os.path import devnull
from shutil import which
from typing import TYPE_CHECKING, Literal
from packaging.version import Version
from ..settings import settings
class _WebdriverState:
    reuse: bool
    kind: DriverKind | None
    current: WebDriver | None
    _drivers: set[WebDriver]

    def __init__(self, *, kind: DriverKind | None=None, reuse: bool=True) -> None:
        self.kind = kind
        self.reuse = reuse
        self.current = None
        self._drivers = set()

    def terminate(self, driver: WebDriver) -> None:
        self._drivers.remove(driver)
        driver.quit()

    def reset(self) -> None:
        if self.current is not None:
            self.terminate(self.current)
            self.current = None

    def get(self, scale_factor: float=1) -> WebDriver:
        if not self.reuse or self.current is None or (not scale_factor_less_than_web_driver_device_pixel_ratio(scale_factor, self.current)):
            self.reset()
            self.current = self.create(scale_factor=scale_factor)
        return self.current

    def create(self, kind: DriverKind | None=None, scale_factor: float=1) -> WebDriver:
        driver = self._create(kind, scale_factor=scale_factor)
        self._drivers.add(driver)
        return driver

    def _create(self, kind: DriverKind | None, scale_factor: float=1) -> WebDriver:
        driver_kind = kind or self.kind
        if driver_kind is None:
            driver = _try_create_chromium_webdriver(scale_factor=scale_factor)
            if driver is not None:
                self.kind = 'chromium'
                return driver
            driver = _try_create_firefox_webdriver(scale_factor=scale_factor)
            if driver is not None:
                self.kind = 'firefox'
                return driver
            raise RuntimeError("Neither firefox and geckodriver nor a variant of chromium browser and chromedriver are available on system PATH. You can install the former with 'conda install -c conda-forge firefox geckodriver'.")
        elif driver_kind == 'chromium':
            return create_chromium_webdriver(scale_factor=scale_factor)
        elif driver_kind == 'firefox':
            return create_firefox_webdriver(scale_factor=scale_factor)
        else:
            raise ValueError(f"'{driver_kind}' is not a recognized webdriver kind")

    def cleanup(self) -> None:
        self.reset()
        for driver in list(self._drivers):
            self.terminate(driver)