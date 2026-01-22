from __future__ import annotations
from typing import Final
from urllib.parse import urljoin
import tornado.web
from streamlit import config, net_util, url_util
def _get_browser_address_bar_port() -> int:
    """Get the app URL that will be shown in the browser's address bar.

    That is, this is the port where static assets will be served from. In dev,
    this is different from the URL that will be used to connect to the
    server-browser websocket.

    """
    if config.get_option('global.developmentMode'):
        return DEVELOPMENT_PORT
    return int(config.get_option('browser.serverPort'))