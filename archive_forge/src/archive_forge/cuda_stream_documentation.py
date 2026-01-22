import logging
import threading
import cupy
from ray.util.collective.collective_group import nccl_util
from ray.util.collective.const import ENV
Initialize the stream pool only for once.