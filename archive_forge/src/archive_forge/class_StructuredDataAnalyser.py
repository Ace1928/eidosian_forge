import numpy as np
import tensorflow as tf
from autokeras.engine import analyser
class StructuredDataAnalyser(InputAnalyser):

    def __init__(self, column_names=None, column_types=None, **kwargs):
        super().__init__(**kwargs)
        self.column_names = column_names
        self.column_types = column_types
        self.count_numerical = None
        self.count_categorical = None
        self.count_unique_numerical = []
        self.num_col = None

    def update(self, data):
        super().update(data)
        if len(self.shape) != 2:
            return
        if data.dtype != tf.string:
            data = tf.strings.as_string(data)
        data = data.numpy()
        for instance in data:
            self._update_instance(instance)

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

    def finalize(self):
        self.check()
        self.infer_column_types()

    def get_input_name(self):
        return 'StructuredDataInput'

    def check(self):
        if len(self.shape) != 2:
            raise ValueError('Expect the data to {input_name} to have shape (batch_size, num_features), but got input shape {shape}.'.format(input_name=self.get_input_name(), shape=self.shape))
        if self.column_names is None:
            if self.column_types:
                raise ValueError('column_names must be specified, if column_types is specified.')
            self.column_names = [str(index) for index in range(self.shape[1])]
        if len(self.column_names) != self.shape[1]:
            raise ValueError('Expect column_names to have length {expect} but got {actual}.'.format(expect=self.shape[1], actual=len(self.column_names)))

    def infer_column_types(self):
        column_types = {}
        for i in range(self.num_col):
            if self.count_categorical[i] > 0:
                column_types[self.column_names[i]] = CATEGORICAL
            elif len(self.count_unique_numerical[i]) / self.count_numerical[i] < 0.05:
                column_types[self.column_names[i]] = CATEGORICAL
            else:
                column_types[self.column_names[i]] = NUMERICAL
        if self.column_types is None:
            self.column_types = {}
        for key, value in column_types.items():
            if key not in self.column_types:
                self.column_types[key] = value