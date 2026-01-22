import re
def _compare_pre_release(self, other):
    """Compare alpha/beta/rc/final."""
    if self.pre_release == other.pre_release:
        vercmp = 0
    elif self.pre_release == 'final':
        vercmp = 1
    elif other.pre_release == 'final':
        vercmp = -1
    elif self.pre_release > other.pre_release:
        vercmp = 1
    else:
        vercmp = -1
    return vercmp