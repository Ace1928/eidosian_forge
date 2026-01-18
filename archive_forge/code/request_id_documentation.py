import re
from oslo_context import context
import webob.dec
from oslo_middleware import base
Middleware that ensures request ID.

    It ensures to assign request ID for each API request and set it to
    request environment. The request ID is also added to API response.
    