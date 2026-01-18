import threading
from datetime import datetime
from time import perf_counter
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
from torch import optim
from torch.testing._internal.dist_utils import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import RpcAgentTestFixture
def timed_log(text):
    print(f'{datetime.now().strftime('%H:%M:%S')} {text}')