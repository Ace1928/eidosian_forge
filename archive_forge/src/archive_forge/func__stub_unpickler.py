import io
import pickle
import warnings
from collections.abc import Collection
from typing import Dict, List, Optional, Set, Tuple, Type, Union
from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE
def _stub_unpickler():
    return 'STUB'