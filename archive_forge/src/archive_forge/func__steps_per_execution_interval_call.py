import logging
import threading
import time
import numpy as np
def _steps_per_execution_interval_call(self):
    while not self.steps_per_execution_stop_event.is_set():
        self._measure_and_tune()
        self.steps_per_execution_stop_event.wait(self.interval)