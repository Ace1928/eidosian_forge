from enum import Enum
import logging
from typing import Iterator, List, Optional, Tuple, cast
import torch
import torch.distributed as dist
from .graph_manager import GraphManager
from .mixing_manager import MixingManager, UniformMixing
class PushSum(Gossiper):
    """1-peer Push-Sum consensus averaging module"""

    def mix(self, out_msg: torch.Tensor, ps_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consensus averaging step"""
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'.format(self.in_edges, self.out_edges))
        mixed_out_msgs = self.mix_out_msg_(out_msg, ps_weight)
        for out_edge in self.out_edges:
            msg = next(mixed_out_msgs)
            assert self.rank == out_edge.src
            req = dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))
        if len(self.in_edges) == 1:
            in_edge = self.in_edges[0]
            dist.broadcast(tensor=self.in_msg_buffer, src=in_edge.src, group=in_edge.process_group)
        else:
            self.in_msg_buffer.zero_()
            for in_edge in self.in_edges:
                dist.broadcast(tensor=self.placeholder, src=in_edge.src, group=in_edge.process_group)
                self.in_msg_buffer.add_(self.placeholder)
        self.refresh_peers_()
        self.clean_msg_buffers_()
        return self.parse_in_msg_buffer()