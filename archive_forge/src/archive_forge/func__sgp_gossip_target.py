from enum import Enum
import functools
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import torch
from torch.autograd import Variable
import torch.distributed as dist
from torch.nn.modules import Module
from .gossiper import Gossiper, PushPull, PushSum
from .graph_manager import GraphManager
from .graph_manager import NPeerDynamicDirectedExponentialGraph as NPDDEGraph
from .mixing_manager import MixingManager, UniformMixing
from .utils import (
from .utils.cuda_metering import EventRecorder, create_event_recorder
@staticmethod
def _sgp_gossip_target(dist_config: Dict[Any, Any], gossip_flag: threading.Event, train_flag: threading.Event, gossip_lock: threading.Lock, gossip_params: List[torch.Tensor], gossip_device_buffer: List[torch.Tensor], gossip_ps_weight: torch.Tensor, gossip_ps_factor: torch.Tensor, gossip_stream: torch.cuda.Stream) -> None:
    """Gossip thread, which performs push-sum on model params"""
    logger = make_logger(dist_config['logical_rank'], dist_config['verbose'])
    gossip_params_by_dtype = group_by_dtype(gossip_params)
    gossip_device_buffer_by_dtype = group_by_dtype(gossip_device_buffer)
    gossipers = {}
    gossiper_class = PushSum if dist_config['push_sum'] else PushPull
    for dtype in gossip_params_by_dtype:
        gossipers[dtype] = gossiper_class(flatten_tensors(gossip_params_by_dtype[dtype]), device=cast(torch.device, dist_config['comm_device']), graph=cast(GraphManager, dist_config['graph']), mixing=cast(MixingManager, dist_config['mixing']), rank=dist_config['process_rank'], world_size=dist_config['logical_world_size'], logger=logger)
    dist_config['gossipers'] = gossipers
    gossip_ps_factor.copy_(gossipers[list(gossipers)[0]].mixing_weights['lo'])
    gossip_flag.set()
    while True:
        train_flag.wait()
        logger.debug('received train-flag')
        try:
            with torch.cuda.stream(gossip_stream):
                for dtype in gossip_params_by_dtype:
                    ps_weight, ps_factor = SlowMoDistributedDataParallel._sgp_gossip_into_receive_buffer(gossip_params_by_dtype[dtype], gossipers[dtype], gossip_device_buffer_by_dtype[dtype], gossip_ps_weight, gossip_lock, dist_config)
                gossip_ps_weight.copy_(ps_weight)
                gossip_ps_factor.copy_(ps_factor)
        except RuntimeError as e:
            logger.warning('received runtime error {}'.format(e))
            for gossiper in gossipers.values():
                gossiper.clean_msg_buffers_()
            gossip_ps_weight.fill_(-1)
        finally:
            gossip_stream.synchronize()
            train_flag.clear()
            gossip_flag.set()