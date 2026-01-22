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
class DummyEnv:
    """
    A dummy environment that implements the required subset of the OpenAI gym
    interface. It exists only to avoid a dependency on gym for running the
    tests in this file. It is designed to run for a set max number of iterations,
    returning random states and rewards at each step.
    """

    def __init__(self, state_dim=4, num_iters=10, reward_threshold=475.0):
        self.state_dim = state_dim
        self.num_iters = num_iters
        self.iter = 0
        self.reward_threshold = reward_threshold

    def seed(self, manual_seed):
        torch.manual_seed(manual_seed)

    def reset(self):
        self.iter = 0
        return torch.randn(self.state_dim)

    def step(self, action):
        self.iter += 1
        state = torch.randn(self.state_dim)
        reward = torch.rand(1).item() * self.reward_threshold
        done = self.iter >= self.num_iters
        info = {}
        return (state, reward, done, info)