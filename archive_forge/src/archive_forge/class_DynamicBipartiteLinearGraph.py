from abc import ABC, abstractmethod
from math import log as mlog
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist
class DynamicBipartiteLinearGraph(GraphManager):

    def _make_graph(self) -> None:
        for rank in range(self.world_size):
            for i in range(1, self.world_size):
                f_peer = self._rotate_forward(rank, i)
                b_peer = self._rotate_backward(rank, i)
                if not self.is_passive(rank) and (self.is_passive(f_peer) and self.is_passive(b_peer)):
                    self._add_peers(rank, [f_peer, b_peer])
                elif self.is_passive(rank) and (not (self.is_passive(f_peer) or self.is_passive(b_peer))):
                    self._add_peers(rank, [f_peer, b_peer])

    def is_regular_graph(self) -> bool:
        return True

    def is_bipartite_graph(self) -> bool:
        return True

    def is_passive(self, rank: Optional[int]=None) -> bool:
        rank = self.rank if rank is None else rank
        return rank % 2 == 0

    def is_dynamic_graph(self) -> bool:
        return True