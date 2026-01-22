import typing
from selenium.webdriver import DesiredCapabilities
from selenium.webdriver.chromium.remote_connection import ChromiumRemoteConnection
class ChromeRemoteConnection(ChromiumRemoteConnection):
    browser_name = DesiredCapabilities.CHROME['browserName']

    def __init__(self, remote_server_addr: str, keep_alive: bool=True, ignore_proxy: typing.Optional[bool]=False) -> None:
        super().__init__(remote_server_addr=remote_server_addr, vendor_prefix='goog', browser_name=ChromeRemoteConnection.browser_name, keep_alive=keep_alive, ignore_proxy=ignore_proxy)