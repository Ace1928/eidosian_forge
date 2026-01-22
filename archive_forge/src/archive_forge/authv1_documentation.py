import datetime
import json
import time
from urllib.parse import urljoin
from keystoneauth1 import discover
from keystoneauth1 import plugin
from keystoneauth1 import exceptions
from keystoneauth1 import loading
from keystoneauth1.identity import base
Return the list of parameters associated with the auth plugin.

        This list may be used to generate CLI or config arguments.
        