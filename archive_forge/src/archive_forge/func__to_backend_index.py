import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple, Union
from torchgen.model import (
from torchgen.utils import assert_never
def _to_backend_index(self) -> BackendIndex:
    """
        WARNING: this will be deprecated once all the codegen places know how to handle ETKernelIndex.
        """
    index: Dict[OperatorName, BackendMetadata] = {}
    for op in self.index:
        kernel_dict = self.index[op]
        assert len(kernel_dict.values()) == 1, f"Can't convert ETKernelIndex to BackendIndex because {op} has more than one kernels. Got {kernel_dict}"
        index[op] = kernel_dict.get(ETKernelKey(default=True), BackendMetadata(kernel='', structured=False, cpp_namespace=''))
    return BackendIndex(dispatch_key=DispatchKey.CPU, use_out_as_primary=False, device_guard=False, external=False, index=index)