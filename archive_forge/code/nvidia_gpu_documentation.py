import re
import os
import logging
from typing import Optional, List, Tuple
import ray._private.thirdparty.pynvml as pynvml
from ray._private.accelerators.accelerator import AcceleratorManager
Nvidia GPU accelerators.