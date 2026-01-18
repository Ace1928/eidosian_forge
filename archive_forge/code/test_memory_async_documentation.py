import asyncio
import gc
import shutil
import pytest
from joblib.memory import (AsyncMemorizedFunc, AsyncNotMemorizedFunc,
from joblib.test.common import np, with_numpy
from joblib.testing import raises
from .test_memory import (corrupt_single_cache_item,
Check that mmap_mode is respected even at the first call