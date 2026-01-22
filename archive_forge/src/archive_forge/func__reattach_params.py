from typing import Any, Callable, List, Optional, Union
import torch
@torch.no_grad()
def _reattach_params(self) -> None:
    """
        Given the parameters which have been registered previously, rebuild the whole bucket
        """
    assert len(self._params) > 0
    self._fill = 0
    for p in self._params:
        if p.dtype != self.buffer.dtype:
            p.data = p.data.to(self.buffer.dtype)
        self._add_param_as_view(p, keep_existing_value=False)