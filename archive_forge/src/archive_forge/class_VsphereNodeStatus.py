import ipaddress
import time
from datetime import datetime
from enum import Enum
class VsphereNodeStatus(Enum):
    CREATING = 'creating'
    CREATED = 'created'