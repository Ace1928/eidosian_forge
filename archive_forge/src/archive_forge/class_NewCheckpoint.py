import json
from typing import Any, List
from uuid import uuid4
from fs.base import FS as FSBase
from triad import assert_or_throw
class NewCheckpoint:
    """A helper class for adding new checkpoints

    :param checkpoint: the parent checkpoint

    .. attention::

        Do not construct this class directly, please read
        :ref:`Checkpoint Tutorial </notebooks/checkpoint.ipynb>`
        for details
    """

    def __init__(self, checkpoint: Checkpoint):
        self._parent = checkpoint
        self._name = str(uuid4())

    def __enter__(self) -> FSBase:
        return self._parent._fs.makedir(self._name)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_traceback: Any) -> None:
        if exc_type is not None:
            try:
                self._parent._fs.removetree(self._name)
            except Exception:
                pass
        else:
            new_iterations = self._parent._iterations + [self._name]
            self._parent._fs.writetext(_CHECKPOINT_STATE_FILE, json.dumps(new_iterations))
            self._parent._iterations = new_iterations