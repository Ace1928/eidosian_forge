import logging
import os
import ray
from ray._private.ray_constants import LOGGER_FORMAT, LOGGER_LEVEL

        Returns the underlying Logger, with the `propagate` attribute set
        to the same value as `log_to_stdout`. For example, when
        `log_to_stdout = False`, we do not want the `DatasetLogger` to
        propagate up to the base Logger which writes to stdout.

        This is a workaround needed due to the DatasetLogger wrapper object
        not having access to the log caller's scope in Python <3.8.
        In the future, with Python 3.8 support, we can use the `stacklevel` arg,
        which allows the logger to fetch the correct calling file/line and
        also removes the need for this getter method:
        `logger.info(msg="Hello world", stacklevel=2)`
        