from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client

        Brackets around IPv6 addresses are stripped in the host field. The host
        field is then exported with brackets in the output of
        L{client.URI.toBytes}.
        