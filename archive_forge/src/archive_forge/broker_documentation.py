import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop

    Redis SSL class offering connection to the broker over SSL.
    This does not currently support SSL settings through the url, only through
    the broker_use_ssl celery configuration.
    