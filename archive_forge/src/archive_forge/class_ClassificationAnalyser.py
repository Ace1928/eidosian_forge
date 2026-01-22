import numpy as np
from autokeras.engine import analyser
class ClassificationAnalyser(TargetAnalyser):

    def __init__(self, num_classes=None, multi_label=False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.label_encoder = None
        self.multi_label = multi_label
        self.labels = set()

    def update(self, data):
        super().update(data)
        if len(self.shape) > 2:
            raise ValueError('Expect the target data for {name} to have shape (batch_size, num_classes), but got {shape}.'.format(name=self.name, shape=self.shape))
        if len(self.shape) > 1 and self.shape[1] > 1:
            return
        self.labels = self.labels.union(set(np.unique(data.numpy())))

    def finalize(self):
        self.labels = sorted(list(self.labels))
        if not self.num_classes:
            if self.encoded:
                if len(self.shape) == 1 or self.shape[1:] == [1]:
                    self.num_classes = 2
                else:
                    self.num_classes = self.shape[1]
            else:
                self.num_classes = len(self.labels)
        if self.num_classes < 2:
            raise ValueError('Expect the target data for {name} to have at least 2 classes, but got {num_classes}.'.format(name=self.name, num_classes=self.num_classes))
        expected = self.get_expected_shape()
        actual = self.shape[1:]
        if len(actual) == 0:
            actual = [1]
        if self.encoded and actual != expected:
            raise ValueError('Expect the target data for {name} to have shape {expected}, but got {actual}.'.format(name=self.name, expected=expected, actual=self.shape[1:]))

    def get_expected_shape(self):
        if self.num_classes == 2 and (not self.multi_label):
            return [1]
        return [self.num_classes]

    @property
    def encoded(self):
        return self.encoded_for_sigmoid or self.encoded_for_softmax

    @property
    def encoded_for_sigmoid(self):
        if len(self.labels) != 2:
            return False
        return sorted(self.labels) == [0, 1]

    @property
    def encoded_for_softmax(self):
        return len(self.shape) > 1 and self.shape[1] > 1