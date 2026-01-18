import numpy as np
from itertools import count
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
from torch.testing._internal.dist_utils import dist_init, worker_name
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture

        This function is mostly borrowed from the Reinforcement Learning example.
        See https://github.com/pytorch/examples/tree/master/reinforcement_learning
        The main difference is that it joins all probs and rewards from
        different observers into one list, and uses the minimum observer rewards
        as the reward of the current episode.
        