import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn

        Automatically wrap child modules of *module* that meet the given
        criteria with :func:`auto_wrap`.

        Args:
            module (nn.Module):
                module to recursively wrap
            auto_wrap_policy (Callable, Optional):
                optionally, override the :func:`auto_wrap_policy` from the context.

        Returns:
            (nn.Module, int):
                Wrapped module and the number parameters wrapped recursively.
        