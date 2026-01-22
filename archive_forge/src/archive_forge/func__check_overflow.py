from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional
import torch
def _check_overflow(self, grad_norm: float) -> None:
    if grad_norm == float('inf') or grad_norm != grad_norm:
        prev_scale = self.loss_scale
        iter_since_rescale = self._iter - self._last_rescale_iter
        self._last_overflow_iter = self._iter
        self._overflows_since_rescale += 1
        pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
        if pct_overflow >= self.tolerance:
            self._decrease_loss_scale()
            self._last_rescale_iter = self._iter
            self._overflows_since_rescale = 0
        if self.loss_scale <= self.min_loss_scale:
            self.loss_scale = prev_scale
            raise FloatingPointError('Minimum loss scale reached ({}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.'.format(self.min_loss_scale))
        self._iter += 1
        raise OverflowError('setting loss scale to: ' + str(self.loss_scale))