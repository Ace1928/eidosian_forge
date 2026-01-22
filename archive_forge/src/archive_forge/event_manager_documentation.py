import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
Wait up to ``timeout`` seconds for ``num_results`` futures to resolve.

        If ``timeout=None``, this method will block until all `num_results`` futures
        resolve. If ``num_results=None``, this method will await all tracked futures.

        For every future that resolves, the respective associated callbacks will be
        invoked.

        Args:
            timeout: Timeout in second to wait for futures to resolve.
            num_results: Number of futures to await. If ``None``, will wait for
                all tracked futures to resolve.

        