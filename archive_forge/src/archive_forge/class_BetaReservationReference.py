import collections
import re
import sys
from typing import Any, Optional, Tuple, Type, Union
from absl import app
from pyglib import stringutil
class BetaReservationReference(ReservationReference):
    """Reference for v1beta1 reservation service."""