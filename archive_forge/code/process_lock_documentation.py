import errno
import logging
import os
import threading
import time
import six
from fasteners import _utils
Checks if the path that this lock exists at actually exists.