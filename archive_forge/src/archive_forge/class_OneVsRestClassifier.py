import array
import itertools
import warnings
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from .base import (
from .metrics.pairwise import pairwise_distances_argmin
from .preprocessing import LabelBinarizer
from .utils import check_random_state
from .utils._param_validation import HasMethods, Interval
from .utils._tags import _safe_tags
from .utils.metadata_routing import (
from .utils.metaestimators import _safe_split, available_if
from .utils.multiclass import (
from .utils.parallel import Parallel, delayed
from .utils.validation import _check_method_params, _num_samples, check_is_fitted
class OneVsRestClassifier(MultiOutputMixin, ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """One-vs-the-rest (OvR) multiclass strategy.

    Also known as one-vs-all, this strategy consists in fitting one classifier
    per class. For each classifier, the class is fitted against all the other
    classes. In addition to its computational efficiency (only `n_classes`
    classifiers are needed), one advantage of this approach is its
    interpretability. Since each class is represented by one and one classifier
    only, it is possible to gain knowledge about the class by inspecting its
    corresponding classifier. This is the most commonly used strategy for
    multiclass classification and is a fair default choice.

    OneVsRestClassifier can also be used for multilabel classification. To use
    this feature, provide an indicator matrix for the target `y` when calling
    `.fit`. In other words, the target labels should be formatted as a 2D
    binary (0/1) matrix, where [i, j] == 1 indicates the presence of label j
    in sample i. This estimator uses the binary relevance method to perform
    multilabel classification, which involves training one binary classifier
    independently for each label.

    Read more in the :ref:`User Guide <ovr_classification>`.

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements :term:`fit`.
        When a classifier is passed, :term:`decision_function` will be used
        in priority and it will fallback to :term:`predict_proba` if it is not
        available.
        When a regressor is passed, :term:`predict` is used.

    n_jobs : int, default=None
        The number of jobs to use for the computation: the `n_classes`
        one-vs-rest problems are computed in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: 0.20
           `n_jobs` default changed from 1 to None

    verbose : int, default=0
        The verbosity level, if non zero, progress messages are printed.
        Below 50, the output is sent to stderr. Otherwise, the output is sent
        to stdout. The frequency of the messages increases with the verbosity
        level, reporting all iterations at 10. See :class:`joblib.Parallel` for
        more details.

        .. versionadded:: 1.1

    Attributes
    ----------
    estimators_ : list of `n_classes` estimators
        Estimators used for predictions.

    classes_ : array, shape = [`n_classes`]
        Class labels.

    n_classes_ : int
        Number of classes.

    label_binarizer_ : LabelBinarizer object
        Object used to transform multiclass labels to binary labels and
        vice-versa.

    multilabel_ : boolean
        Whether a OneVsRestClassifier is a multilabel classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    OneVsOneClassifier : One-vs-one multiclass strategy.
    OutputCodeClassifier : (Error-Correcting) Output-Code multiclass strategy.
    sklearn.multioutput.MultiOutputClassifier : Alternate way of extending an
        estimator for multilabel classification.
    sklearn.preprocessing.MultiLabelBinarizer : Transform iterable of iterables
        to binary indicator matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.svm import SVC
    >>> X = np.array([
    ...     [10, 10],
    ...     [8, 10],
    ...     [-5, 5.5],
    ...     [-5.4, 5.5],
    ...     [-20, -20],
    ...     [-15, -20]
    ... ])
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> clf = OneVsRestClassifier(SVC()).fit(X, y)
    >>> clf.predict([[-19, -20], [9, 9], [-5, 5]])
    array([2, 0, 1])
    """
    _parameter_constraints = {'estimator': [HasMethods(['fit'])], 'n_jobs': [Integral, None], 'verbose': ['verbose']}

    def __init__(self, estimator, *, n_jobs=None, verbose=0):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        **fit_params : dict
            Parameters passed to the ``estimator.fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        _raise_for_params(fit_params, self, 'fit')
        routed_params = process_routing(self, 'fit', **fit_params)
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)((delayed(_fit_binary)(self.estimator, X, column, fit_params=routed_params.estimator.fit, classes=['not %s' % self.label_binarizer_.classes_[i], self.label_binarizer_.classes_[i]]) for i, column in enumerate(columns)))
        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], 'feature_names_in_'):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_
        return self

    @available_if(_estimators_has('partial_fit'))
    @_fit_context(prefer_skip_nested_validation=False)
    def partial_fit(self, X, y, classes=None, **partial_fit_params):
        """Partially fit underlying estimators.

        Should be used when memory is inefficient to train all data.
        Chunks of data can be passed in several iterations.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        classes : array, shape (n_classes, )
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is only required in the first call of partial_fit
            and can be omitted in the subsequent calls.

        **partial_fit_params : dict
            Parameters passed to the ``estimator.partial_fit`` method of each
            sub-estimator.

            .. versionadded:: 1.4
                Only available if `enable_metadata_routing=True`. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        self : object
            Instance of partially fitted estimator.
        """
        _raise_for_params(partial_fit_params, self, 'partial_fit')
        routed_params = process_routing(self, 'partial_fit', **partial_fit_params)
        if _check_partial_fit_first_call(self, classes):
            self.estimators_ = [clone(self.estimator) for _ in range(self.n_classes_)]
            self.label_binarizer_ = LabelBinarizer(sparse_output=True)
            self.label_binarizer_.fit(self.classes_)
        if len(np.setdiff1d(y, self.classes_)):
            raise ValueError(('Mini-batch contains {0} while classes ' + 'must be subset of {1}').format(np.unique(y), self.classes_))
        Y = self.label_binarizer_.transform(y)
        Y = Y.tocsc()
        columns = (col.toarray().ravel() for col in Y.T)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)((delayed(_partial_fit_binary)(estimator, X, column, partial_fit_params=routed_params.estimator.partial_fit) for estimator, column in zip(self.estimators_, columns)))
        if hasattr(self.estimators_[0], 'n_features_in_'):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        return self

    def predict(self, X):
        """Predict multi-class targets using underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data.

        Returns
        -------
        y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_classes)
            Predicted multi-class targets.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == 'multiclass':
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary(e, X)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[argmaxima]
        else:
            thresh = _threshold_for_binary_predict(self.estimators_[0])
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary(e, X) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr), shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)

    @available_if(_estimators_has('predict_proba'))
    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by label of classes.

        Note that in the multilabel case, each sample can have any number of
        labels. This returns the marginal probability that the given sample has
        the label in question. For example, it is entirely consistent that two
        labels both have a 90% probability of applying to a given sample.

        In the single label multiclass case, the rows of the returned matrix
        sum to 1.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)
        Y = np.array([e.predict_proba(X)[:, 1] for e in self.estimators_]).T
        if len(self.estimators_) == 1:
            Y = np.concatenate((1 - Y, Y), axis=1)
        if not self.multilabel_:
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
        return Y

    @available_if(_estimators_has('decision_function'))
    def decision_function(self, X):
        """Decision function for the OneVsRestClassifier.

        Return the distance of each sample from the decision boundary for each
        class. This can only be used with estimators which implement the
        `decision_function` method.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes) or (n_samples,) for             binary classification.
            Result of calling `decision_function` on the final estimator.

            .. versionchanged:: 0.19
                output shape changed to ``(n_samples,)`` to conform to
                scikit-learn conventions for binary classification.
        """
        check_is_fitted(self)
        if len(self.estimators_) == 1:
            return self.estimators_[0].decision_function(X)
        return np.array([est.decision_function(X).ravel() for est in self.estimators_]).T

    @property
    def multilabel_(self):
        """Whether this is a multilabel classifier."""
        return self.label_binarizer_.y_type_.startswith('multilabel')

    @property
    def n_classes_(self):
        """Number of classes."""
        return len(self.classes_)

    def _more_tags(self):
        """Indicate if wrapped estimator is using a precomputed Gram matrix"""
        return {'pairwise': _safe_tags(self.estimator, key='pairwise')}

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        .. versionadded:: 1.4

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self.__class__.__name__).add_self_request(self).add(estimator=self.estimator, method_mapping=MethodMapping().add(callee='fit', caller='fit').add(callee='partial_fit', caller='partial_fit'))
        return router