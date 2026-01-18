from selenium.webdriver.chromium.options import ChromiumOptions
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
@use_webview.setter
def use_webview(self, value: bool) -> None:
    self._use_webview = bool(value)