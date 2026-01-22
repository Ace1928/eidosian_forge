import abc
import enum
from taskflow import atom
from taskflow import exceptions as exc
from taskflow.utils import misc
class AlwaysRevertAll(Retry):
    """Retry that always reverts a whole flow."""

    def on_failure(self, **kwargs):
        return REVERT_ALL

    def execute(self, **kwargs):
        pass