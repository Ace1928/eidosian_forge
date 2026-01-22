import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
class LV(PagerCommand):
    """The pager command ``lv``."""

    def command(self) -> List[str]:
        return ['lv']

    def environment_variables(self, config: PagerConfig) -> Optional[Dict[str, str]]:
        if config.color and os.getenv('LV') is None:
            return {'LV': '-c'}
        return None