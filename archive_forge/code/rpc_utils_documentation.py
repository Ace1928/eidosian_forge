import os
import sys
import unittest
from typing import Dict, List, Type
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.ddp_under_dist_autograd_test import (
from torch.testing._internal.distributed.pipe_with_ddp_test import (
from torch.testing._internal.distributed.nn.api.remote_module_test import (
from torch.testing._internal.distributed.rpc.dist_autograd_test import (
from torch.testing._internal.distributed.rpc.dist_optimizer_test import (
from torch.testing._internal.distributed.rpc.jit.dist_autograd_test import (
from torch.testing._internal.distributed.rpc.jit.rpc_test import JitRpcTest
from torch.testing._internal.distributed.rpc.jit.rpc_test_faulty import (
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
from torch.testing._internal.distributed.rpc.faulty_agent_rpc_test import (
from torch.testing._internal.distributed.rpc.rpc_test import (
from torch.testing._internal.distributed.rpc.examples.parameter_server_test import ParameterServerTest
from torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test import (
Mix in the classes needed to autogenerate the tests based on the params.

    Takes a series of test suites, each written against a "generic" agent (i.e.,
    derived from the abstract RpcAgentTestFixture class), as the `tests` args.
    Takes a concrete subclass of RpcAgentTestFixture, which specializes it for a
    certain agent, as the `mixin` arg. Produces all combinations of them.
    Returns a dictionary of class names to class type
    objects which can be inserted into the global namespace of the calling
    module. The name of each test will be a concatenation of the `prefix` arg
    and the original name of the test suite.
    The `module_name` should be the name of the calling module so
    that the classes can be fixed to make it look like they belong to it, which
    is necessary for pickling to work on them.
    