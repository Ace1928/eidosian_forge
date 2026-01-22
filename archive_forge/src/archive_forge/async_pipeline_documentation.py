import logging
import os
from threading import Event
from typing import Dict, List, Optional, Union
import torch
from .async_schedule import AsyncEventLoop, ModuleWrapper
from .messages import MakeTransport
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals
Runs pipeline parallelism.

        It modifies the given batches in place.

        