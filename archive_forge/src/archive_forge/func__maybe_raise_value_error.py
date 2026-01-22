import dataclasses
from typing import Collection
from werkzeug.datastructures import Headers
from werkzeug import http
from tensorboard.util import tb_logging
def _maybe_raise_value_error(error_msg):
    logger.warning('In 3.0, this warning will become an error:\n%s' % error_msg)