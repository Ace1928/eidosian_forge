import os, sys
import re
import logging; log = logging.getLogger(__name__)
from passlib.utils.compat import print_
write fake GAE ``app.yaml`` to current directory so nosegae will work