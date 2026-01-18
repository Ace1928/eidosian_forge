import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
@property
def will_move(self):
    """Whether capability causes cursor movement."""
    return self.name in CAPABILITIES_CAUSE_MOVEMENT