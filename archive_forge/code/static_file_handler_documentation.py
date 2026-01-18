import os
import re
import tornado.web
from .paths import collect_static_paths
A static file handler that 'merges' a list of directories

    If initialized like this::

        application = web.Application([
            (r"/content/(.*)", web.MultiStaticFileHandler, {"paths": ["/var/1", "/var/2"]}),
        ])

    A file will be looked up in /var/1 first, then in /var/2.

    