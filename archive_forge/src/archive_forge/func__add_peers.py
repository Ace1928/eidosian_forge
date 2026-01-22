from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
def _add_peers(self, rank: int, peers: List[int]) -> None:
    for peer in peers:
        if peer not in self.phone_book[rank]:
            self.phone_book[rank].append(Edge(local_master_rank=self.rank * self.nprocs_per_node, dest=peer * self.nprocs_per_node, src=rank * self.nprocs_per_node, local_rank=self.local_rank))