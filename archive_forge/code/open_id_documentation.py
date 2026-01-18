import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
Render the page footer