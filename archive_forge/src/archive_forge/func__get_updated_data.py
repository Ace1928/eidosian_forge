import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
def _get_updated_data(self) -> Dict[str, Any]:
    updated_data = {}
    for attr in self.manager._update_attrs.required:
        updated_data[attr] = getattr(self, attr)
    updated_data.update(self._updated_attrs)
    return updated_data