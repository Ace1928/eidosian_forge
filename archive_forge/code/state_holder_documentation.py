from __future__ import annotations
import datetime
import os
import threading
from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Iterator
Reset the state holder with new blocks. Used during reload mode.