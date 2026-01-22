import json
import logging
import time
from oauth2client import _helpers
from oauth2client import _pure_python_crypt
class AppIdentityError(Exception):
    """Error to indicate crypto failure."""