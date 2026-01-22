from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.utils import io_utils
from keras.src.utils.progbar import Progbar
def _update_progbar(self, batch, logs=None):
    """Updates the progbar."""
    logs = logs or {}
    self._maybe_init_progbar()
    self.seen = batch + 1
    if self.verbose == 1:
        self.progbar.update(self.seen, list(logs.items()), finalize=False)