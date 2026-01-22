from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
class DotNotTrackScope:

    def __enter__(self):
        self.original_value = is_tracking_enabled()
        set_global_attribute('tracking_on', False)

    def __exit__(self, *args, **kwargs):
        set_global_attribute('tracking_on', self.original_value)