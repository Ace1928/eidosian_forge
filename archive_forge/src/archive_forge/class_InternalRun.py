from wandb._globals import _datatypes_set_callback
from .. import wandb_run
class InternalRun(wandb_run.Run):

    def __init__(self, run_obj, settings, datatypes_cb):
        super().__init__(settings=settings)
        self._run_obj = run_obj
        _datatypes_set_callback(datatypes_cb)

    def _set_backend(self, backend):
        pass