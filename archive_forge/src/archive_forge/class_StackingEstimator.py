import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.utils import check_array
class StackingEstimator(BaseEstimator, TransformerMixin):
    """Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        """
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """Transform data by adding two synthetic feature(s).

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """
        X = check_array(X)
        X_transformed = np.copy(X)
        if is_classifier(self.estimator) and hasattr(self.estimator, 'predict_proba'):
            y_pred_proba = self.estimator.predict_proba(X)
            if np.all(np.isfinite(y_pred_proba)):
                X_transformed = np.hstack((y_pred_proba, X))
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))
        return X_transformed