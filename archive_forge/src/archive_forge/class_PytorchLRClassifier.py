from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.

    For examples on standalone use (i.e., non-TPOT) refer to:
    https://github.com/trang1618/tpot-nn/blob/master/tpot_nn/estimator_sandbox.py
    """

    def __init__(self, num_epochs=10, batch_size=16, learning_rate=0.02, weight_decay=0.0001, verbose=False):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.input_size = None
        self.num_classes = None
        self.network = None
        self.loss_function = None
        self.optimizer = None
        self.data_loader = None
        self.train_dset_len = None
        self.device = None

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()
        X, y = self.validate_inputs(X, y)
        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        train_dset = TensorDataset(X, y)
        self.network = _LR(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}