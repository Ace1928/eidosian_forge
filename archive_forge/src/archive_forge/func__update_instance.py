import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
def _update_instance(self, x):
    if self.num_col is None:
        self.num_col = len(x)
        self.count_numerical = np.zeros(self.num_col)
        self.count_categorical = np.zeros(self.num_col)
        for _ in range(len(x)):
            self.count_unique_numerical.append({})
    for i in range(self.num_col):
        x[i] = x[i].decode('utf-8')
        try:
            tmp_num = float(x[i])
            self.count_numerical[i] += 1
            if tmp_num not in self.count_unique_numerical[i]:
                self.count_unique_numerical[i][tmp_num] = 1
            else:
                self.count_unique_numerical[i][tmp_num] += 1
        except ValueError:
            self.count_categorical[i] += 1