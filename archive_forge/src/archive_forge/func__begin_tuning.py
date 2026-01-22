import logging
import threading
import time
import numpy as np
def _begin_tuning(self):
    self.start_time = time.time()
    self.init_iterations = self.optimizer.iterations.numpy()
    self.init_spe = self._steps_per_execution.numpy().item()
    self.spe_last_logged = {'iteration': self.init_iterations, 'time_secs': self.start_time}
    self.rgsps = []
    self.avg_rgsps = 0
    self.prev_avg_rgsps = 0
    self.spe_tune_last_action_add = True
    self.spe_measurement_count = 0