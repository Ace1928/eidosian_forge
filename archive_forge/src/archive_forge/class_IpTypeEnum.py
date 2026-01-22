from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
class IpTypeEnum(enum.Enum):
    """Enum representing the ip type of instances."""
    INTERNAL = 'internal ip address'
    EXTERNAL = 'external ip address'