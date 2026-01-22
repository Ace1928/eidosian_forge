from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted
from sklearn.utils.multiclass import type_of_target
class PytorchClassifier(PytorchEstimator, ClassifierMixin):

    @abstractmethod
    def _init_model(self, X, y):
        pass

    def fit(self, X, y):
        """Generalizable method for fitting a PyTorch estimator to a training
        set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self
            Fitted estimator.
        """
        self._init_model(X, y)
        assert _pytorch_model_is_fully_initialized(self)
        for epoch in range(self.num_epochs):
            for i, (samples, labels) in enumerate(self.data_loader):
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.network(samples)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                if self.verbose and (i + 1) % 100 == 0:
                    print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (epoch + 1, self.num_epochs, i + 1, self.train_dset_len // self.batch_size, loss.item()))
        self.is_fitted_ = True
        return self

    def validate_inputs(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, allow_nd=False)
        assert_all_finite(X)
        assert_all_finite(y)
        if type_of_target(y) != 'binary':
            raise ValueError('Non-binary targets not supported')
        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError('Complex data not supported')
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except (TypeError, ValueError):
                raise ValueError('argument must be a string.* number')
        return (X, y)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.network(rows)
            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)
        return predictions.reshape(-1, 1)

    def transform(self, X):
        return self.predict(X)