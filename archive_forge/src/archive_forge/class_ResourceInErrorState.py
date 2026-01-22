from datetime import datetime
from oslo_utils import timeutils
class ResourceInErrorState(Exception):
    """When resource is in Error state"""

    def __init__(self, obj, fault_msg):
        msg = "'%s' resource is in the error state" % obj.__class__.__name__
        if fault_msg:
            msg += " due to '%s'" % fault_msg
        self.message = '%s.' % msg

    def __str__(self):
        return self.message