from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
def _init_model(self, X, y):
    device = _get_cuda_device_if_available()
    X, y = self.validate_inputs(X, y)
    self.input_size = X.shape[-1]
    self.num_classes = len(set(y))
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    train_dset = TensorDataset(X, y)
    self.network = _MLP(self.input_size, self.num_classes).to(device)
    self.loss_function = nn.CrossEntropyLoss()
    self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    self.data_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2)
    self.train_dset_len = len(train_dset)
    self.device = device