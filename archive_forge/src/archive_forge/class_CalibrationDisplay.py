import warnings
from inspect import signature
from math import log
from numbers import Integral, Real
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import Bunch
from ._loss import HalfBinomialLoss
from .base import (
from .isotonic import IsotonicRegression
from .model_selection import check_cv, cross_val_predict
from .preprocessing import LabelEncoder, label_binarize
from .svm import LinearSVC
from .utils import (
from .utils._param_validation import (
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils._response import _get_response_values, _process_predict_proba
from .utils.metadata_routing import (
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import (
class CalibrationDisplay(_BinaryClassifierCurveDisplayMixin):
    """Calibration curve (also known as reliability diagram) visualization.

    It is recommended to use
    :func:`~sklearn.calibration.CalibrationDisplay.from_estimator` or
    :func:`~sklearn.calibration.CalibrationDisplay.from_predictions`
    to create a `CalibrationDisplay`. All parameters are stored as attributes.

    Read more about calibration in the :ref:`User Guide <calibration>` and
    more about the scikit-learn visualization API in :ref:`visualizations`.

    .. versionadded:: 1.0

    Parameters
    ----------
    prob_true : ndarray of shape (n_bins,)
        The proportion of samples whose class is the positive class (fraction
        of positives), in each bin.

    prob_pred : ndarray of shape (n_bins,)
        The mean predicted probability in each bin.

    y_prob : ndarray of shape (n_samples,)
        Probability estimates for the positive class, for each sample.

    estimator_name : str, default=None
        Name of estimator. If None, the estimator name is not shown.

    pos_label : int, float, bool or str, default=None
        The positive class when computing the calibration curve.
        By default, `pos_label` is set to `estimators.classes_[1]` when using
        `from_estimator` and set to 1 when using `from_predictions`.

        .. versionadded:: 1.1

    Attributes
    ----------
    line_ : matplotlib Artist
        Calibration curve.

    ax_ : matplotlib Axes
        Axes with calibration curve.

    figure_ : matplotlib Figure
        Figure containing the curve.

    See Also
    --------
    calibration_curve : Compute true and predicted probabilities for a
        calibration curve.
    CalibrationDisplay.from_predictions : Plot calibration curve using true
        and predicted labels.
    CalibrationDisplay.from_estimator : Plot calibration curve using an
        estimator and data.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.calibration import calibration_curve, CalibrationDisplay
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> clf = LogisticRegression(random_state=0)
    >>> clf.fit(X_train, y_train)
    LogisticRegression(random_state=0)
    >>> y_prob = clf.predict_proba(X_test)[:, 1]
    >>> prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    >>> disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
    >>> disp.plot()
    <...>
    """

    def __init__(self, prob_true, prob_pred, y_prob, *, estimator_name=None, pos_label=None):
        self.prob_true = prob_true
        self.prob_pred = prob_pred
        self.y_prob = y_prob
        self.estimator_name = estimator_name
        self.pos_label = pos_label

    def plot(self, *, ax=None, name=None, ref_line=True, **kwargs):
        """Plot visualization.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Parameters
        ----------
        ax : Matplotlib Axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        name : str, default=None
            Name for labeling curve. If `None`, use `estimator_name` if
            not `None`, otherwise no labeling is shown.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`
            Object that stores computed values.
        """
        self.ax_, self.figure_, name = self._validate_plot_params(ax=ax, name=name)
        info_pos_label = f'(Positive class: {self.pos_label})' if self.pos_label is not None else ''
        line_kwargs = {'marker': 's', 'linestyle': '-'}
        if name is not None:
            line_kwargs['label'] = name
        line_kwargs.update(**kwargs)
        ref_line_label = 'Perfectly calibrated'
        existing_ref_line = ref_line_label in self.ax_.get_legend_handles_labels()[1]
        if ref_line and (not existing_ref_line):
            self.ax_.plot([0, 1], [0, 1], 'k:', label=ref_line_label)
        self.line_ = self.ax_.plot(self.prob_pred, self.prob_true, **line_kwargs)[0]
        self.ax_.legend(loc='lower right')
        xlabel = f'Mean predicted probability {info_pos_label}'
        ylabel = f'Fraction of positives {info_pos_label}'
        self.ax_.set(xlabel=xlabel, ylabel=ylabel)
        return self

    @classmethod
    def from_estimator(cls, estimator, X, y, *, n_bins=5, strategy='uniform', pos_label=None, name=None, ref_line=True, ax=None, **kwargs):
        """Plot calibration curve using a binary classifier and data.

        A calibration curve, also known as a reliability diagram, uses inputs
        from a binary classifier and plots the average predicted probability
        for each bin against the fraction of positive classes, on the
        y-axis.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Read more about calibration in the :ref:`User Guide <calibration>` and
        more about the scikit-learn visualization API in :ref:`visualizations`.

        .. versionadded:: 1.0

        Parameters
        ----------
        estimator : estimator instance
            Fitted classifier or a fitted :class:`~sklearn.pipeline.Pipeline`
            in which the last estimator is a classifier. The classifier must
            have a :term:`predict_proba` method.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input values.

        y : array-like of shape (n_samples,)
            Binary target values.

        n_bins : int, default=5
            Number of bins to discretize the [0, 1] interval into when
            calculating the calibration curve. A bigger number requires more
            data.

        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.

            - `'uniform'`: The bins have identical widths.
            - `'quantile'`: The bins have the same number of samples and depend
              on predicted probabilities.

        pos_label : int, float, bool or str, default=None
            The positive class when computing the calibration curve.
            By default, `estimators.classes_[1]` is considered as the
            positive class.

            .. versionadded:: 1.1

        name : str, default=None
            Name for labeling curve. If `None`, the name of the estimator is
            used.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`.
            Object that stores computed values.

        See Also
        --------
        CalibrationDisplay.from_predictions : Plot calibration curve using true
            and predicted labels.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.calibration import CalibrationDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> disp = CalibrationDisplay.from_estimator(clf, X_test, y_test)
        >>> plt.show()
        """
        y_prob, pos_label, name = cls._validate_and_get_response_values(estimator, X, y, response_method='predict_proba', pos_label=pos_label, name=name)
        return cls.from_predictions(y, y_prob, n_bins=n_bins, strategy=strategy, pos_label=pos_label, name=name, ref_line=ref_line, ax=ax, **kwargs)

    @classmethod
    def from_predictions(cls, y_true, y_prob, *, n_bins=5, strategy='uniform', pos_label=None, name=None, ref_line=True, ax=None, **kwargs):
        """Plot calibration curve using true labels and predicted probabilities.

        Calibration curve, also known as reliability diagram, uses inputs
        from a binary classifier and plots the average predicted probability
        for each bin against the fraction of positive classes, on the
        y-axis.

        Extra keyword arguments will be passed to
        :func:`matplotlib.pyplot.plot`.

        Read more about calibration in the :ref:`User Guide <calibration>` and
        more about the scikit-learn visualization API in :ref:`visualizations`.

        .. versionadded:: 1.0

        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True labels.

        y_prob : array-like of shape (n_samples,)
            The predicted probabilities of the positive class.

        n_bins : int, default=5
            Number of bins to discretize the [0, 1] interval into when
            calculating the calibration curve. A bigger number requires more
            data.

        strategy : {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.

            - `'uniform'`: The bins have identical widths.
            - `'quantile'`: The bins have the same number of samples and depend
              on predicted probabilities.

        pos_label : int, float, bool or str, default=None
            The positive class when computing the calibration curve.
            By default `pos_label` is set to 1.

            .. versionadded:: 1.1

        name : str, default=None
            Name for labeling curve.

        ref_line : bool, default=True
            If `True`, plots a reference line representing a perfectly
            calibrated classifier.

        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.

        **kwargs : dict
            Keyword arguments to be passed to :func:`matplotlib.pyplot.plot`.

        Returns
        -------
        display : :class:`~sklearn.calibration.CalibrationDisplay`.
            Object that stores computed values.

        See Also
        --------
        CalibrationDisplay.from_estimator : Plot calibration curve using an
            estimator and data.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.calibration import CalibrationDisplay
        >>> X, y = make_classification(random_state=0)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, random_state=0)
        >>> clf = LogisticRegression(random_state=0)
        >>> clf.fit(X_train, y_train)
        LogisticRegression(random_state=0)
        >>> y_prob = clf.predict_proba(X_test)[:, 1]
        >>> disp = CalibrationDisplay.from_predictions(y_test, y_prob)
        >>> plt.show()
        """
        pos_label_validated, name = cls._validate_from_predictions_params(y_true, y_prob, sample_weight=None, pos_label=pos_label, name=name)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy, pos_label=pos_label)
        disp = cls(prob_true=prob_true, prob_pred=prob_pred, y_prob=y_prob, estimator_name=name, pos_label=pos_label_validated)
        return disp.plot(ax=ax, ref_line=ref_line, **kwargs)