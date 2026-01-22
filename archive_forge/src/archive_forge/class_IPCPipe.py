import concurrent.futures
import json
import multiprocessing.connection
from typing import Any, List, Optional, Union
import torch
import torch.distributed as dist
import torch.multiprocessing.reductions
class IPCPipe:

    def __init__(self, connection, my_device) -> None:
        self.connection = connection
        self.my_device = my_device

    def send(self, tensor: torch.Tensor) -> None:
        assert self.connection is not None, 'Sending to myself!'
        assert tensor.device == self.my_device, f'tensor.device={tensor.device!r} != self.my_device={self.my_device!r}'
        self.connection.send(_serialize_cuda_tensor(tensor))

    def recv(self) -> torch.Tensor:
        assert self.connection is not None, 'Receiving from myself!'
        return _deserialize_cuda_tensor(self.connection.recv(), self.my_device)