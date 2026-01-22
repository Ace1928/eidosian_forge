import time
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
class DummyWaiter:
    """A no-op waiter that simply returns the item being waited on.

    No API call will be made with this waiter; the function returns
    immediately. This waiter is useful for waiting on resource instances in
    check mode, for example.
    """

    def wait(self, definition: Dict, timeout: int, sleep: int, label_selectors: Optional[List[str]]=None) -> Tuple[bool, Optional[Dict], int]:
        return (True, definition, 0)