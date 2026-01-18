import math
import os
import sys
import time
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.utils import io_utils
Estimate the duration of a single step.

        Given the step number `current` and the corresponding time `now` this
        function returns an estimate for how long a single step takes. If this
        is called before one step has been completed (i.e. `current == 0`) then
        zero is given as an estimate. The duration estimate ignores the duration
        of the (assumed to be non-representative) first step for estimates when
        more steps are available (i.e. `current>1`).

        Args:
            current: Index of current step.
            now: The current time.

        Returns: Estimate of the duration of a single step.
        