from oslo_config import cfg
from oslo_middleware import base
Parses RFC7239 Forward headers.

        e.g. for=192.0.2.60;proto=http, for=192.0.2.60;by=203.0.113.43

        