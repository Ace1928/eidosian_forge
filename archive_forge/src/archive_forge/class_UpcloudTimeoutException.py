import json
import time
from libcloud.common.types import LibcloudError
from libcloud.common.exceptions import BaseHTTPError
class UpcloudTimeoutException(LibcloudError):
    pass