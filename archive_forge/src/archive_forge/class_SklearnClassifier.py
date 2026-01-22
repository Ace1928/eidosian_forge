from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist
class SklearnClassifier(ClassifierI):
    """Wrapper for scikit-learn classifiers."""

    def __init__(self, estimator, dtype=float, sparse=True):
        """
        :param estimator: scikit-learn classifier object.

        :param dtype: data type used when building feature array.
            scikit-learn estimators work exclusively on numeric data. The
            default value should be fine for almost all situations.

        :param sparse: Whether to use sparse matrices internally.
            The estimator must support these; not all scikit-learn classifiers
            do (see their respective documentation and look for "sparse
            matrix"). The default value is True, since most NLP problems
            involve sparse feature sets. Setting this to False may take a
            great amount of memory.
        :type sparse: boolean.
        """
        self._clf = estimator
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse)

    def __repr__(self):
        return '<SklearnClassifier(%r)>' % self._clf

    def classify_many(self, featuresets):
        """Classify a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :return: The predicted class label for each input sample.
        :rtype: list
        """
        X = self._vectorizer.transform(featuresets)
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)]

    def prob_classify_many(self, featuresets):
        """Compute per-class probabilities for a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :rtype: list of ``ProbDistI``
        """
        X = self._vectorizer.transform(featuresets)
        y_proba_list = self._clf.predict_proba(X)
        return [self._make_probdist(y_proba) for y_proba in y_proba_list]

    def labels(self):
        """The class labels used by this classifier.

        :rtype: list
        """
        return list(self._encoder.classes_)

    def train(self, labeled_featuresets):
        """
        Train (fit) the scikit-learn estimator.

        :param labeled_featuresets: A list of ``(featureset, label)``
            where each ``featureset`` is a dict mapping strings to either
            numbers, booleans or strings.
        """
        X, y = list(zip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)
        return self

    def _make_probdist(self, y_proba):
        classes = self._encoder.classes_
        return DictionaryProbDist({classes[i]: p for i, p in enumerate(y_proba)})