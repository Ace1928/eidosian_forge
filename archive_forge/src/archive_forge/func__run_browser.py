import json
import websocket
import threading
from parlai.core.params import ParlaiParser
from parlai.scripts.interactive_web import WEB_HTML, STYLE_SHEET, FONT_AWESOME
from http.server import BaseHTTPRequestHandler, HTTPServer
def _run_browser():
    host = opt.get('host', 'localhost')
    serving_port = opt.get('serving_port', 8080)
    httpd = HTTPServer((host, serving_port), BrowserHandler)
    print('Please connect to the link: http://{}:{}/'.format(host, serving_port))
    SHARED['wb'] = httpd
    httpd.serve_forever()