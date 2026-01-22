import html
from urllib.parse import parse_qsl, quote, unquote, urlencode
from paste import request
def _html_extra(self):
    return ''