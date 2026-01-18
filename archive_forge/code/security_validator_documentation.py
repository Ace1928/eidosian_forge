import dataclasses
from typing import Collection
from werkzeug.datastructures import Headers
from werkzeug import http
from tensorboard.util import tb_logging
Initializes an `SecurityValidatorMiddleware`.

        Args:
          application: The WSGI application to wrap (see PEP 3333).
        