import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioOperation(PulseAudioMainloopChild):
    """An asynchronous PulseAudio operation.
    Can be waited for, where it will run until completion or cancellation.
    Remember to `delete()` it with the mainloop lock held, otherwise
    it will be leaked.
    """
    _state_name = {pa.PA_OPERATION_RUNNING: 'Running', pa.PA_OPERATION_DONE: 'Done', pa.PA_OPERATION_CANCELLED: 'Cancelled'}

    def __init__(self, callback_lump, pa_operation: pa.pa_operation) -> None:
        context = callback_lump.context
        assert context.mainloop is not None
        assert pa_operation is not None
        context.check_ptr_not_null(pa_operation)
        super().__init__(context.mainloop)
        self.callback_lump = callback_lump
        self._pa_operation = pa_operation

    def _get_state(self) -> None:
        assert self._pa_operation is not None
        return pa.pa_operation_get_state(self._pa_operation)

    def delete(self) -> None:
        """Unref and delete the operation."""
        if self._pa_operation is not None:
            pa.pa_operation_unref(self._pa_operation)
            self._pa_operation = None
            self.callback_lump = None
            self.context = None

    def cancel(self):
        """Cancel the operation."""
        assert self._pa_operation is not None
        pa.pa_operation_cancel(self._pa_operation)
        return self

    def wait(self):
        """Wait until the operation is either done or cancelled."""
        while self.is_running:
            self.mainloop.wait()
        return self

    @property
    def is_running(self) -> bool:
        return self._get_state() == pa.PA_OPERATION_RUNNING

    @property
    def is_done(self) -> bool:
        return self._get_state() == pa.PA_OPERATION_DONE

    @property
    def is_cancelled(self) -> bool:
        return self._get_state() == pa.PA_OPERATION_CANCELLED