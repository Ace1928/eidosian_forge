from typing import Any, Callable, List, Optional, Tuple
def _check_completion(self):
    if self._completed:
        return
    if self.num_results >= self._max_results:
        self._completed = True
        if self._on_completion:
            self._on_completion(self)