import os
import sys
import threading
from . import process
from . import reduction
class ForkContext(BaseContext):
    _name = 'fork'
    Process = ForkProcess