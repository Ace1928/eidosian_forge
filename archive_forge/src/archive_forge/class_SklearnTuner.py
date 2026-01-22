import collections
import inspect
import os
import pickle
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import base_tuner
@keras_tuner_export(['keras_tuner.SklearnTuner', 'keras_tuner.tuners.SklearnTuner'])
class SklearnTuner(base_tuner.BaseTuner):
    """Tuner for Scikit-learn Models.

    Performs cross-validated hyperparameter search for Scikit-learn models.

    Examples:

    ```python
    import keras_tuner
    from sklearn import ensemble
    from sklearn import datasets
    from sklearn import linear_model
    from sklearn import metrics
    from sklearn import model_selection

    def build_model(hp):
      model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
      if model_type == 'random_forest':
        model = ensemble.RandomForestClassifier(
            n_estimators=hp.Int('n_estimators', 10, 50, step=10),
            max_depth=hp.Int('max_depth', 3, 10))
      else:
        model = linear_model.RidgeClassifier(
            alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
      return model

    tuner = keras_tuner.tuners.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimizationOracle(
            objective=keras_tuner.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory='.',
        project_name='my_project')

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    tuner.search(X_train, y_train)

    best_model = tuner.get_best_models(num_models=1)[0]
    ```

    Args:
        oracle: A `keras_tuner.Oracle` instance. Note that for this `Tuner`,
            the `objective` for the `Oracle` should always be set to
            `Objective('score', direction='max')`. Also, `Oracle`s that exploit
            Neural-Network-specific training (e.g. `Hyperband`) should not be
            used with this `Tuner`.
        hypermodel: A `HyperModel` instance (or callable that takes
            hyperparameters and returns a Model instance).
        scoring: An sklearn `scoring` function. For more information, see
            `sklearn.metrics.make_scorer`. If not provided, the Model's default
            scoring will be used via `model.score`. Note that if you are
            searching across different Model families, the default scoring for
            these Models will often be different. In this case you should
            supply `scoring` here in order to make sure your Models are being
            scored on the same metric.
        metrics: Additional `sklearn.metrics` functions to monitor during
            search. Note that these metrics do not affect the search process.
        cv: An `sklearn.model_selection` Splitter class. Used to
            determine how samples are split up into groups for
            cross-validation.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(self, oracle, hypermodel, scoring=None, metrics=None, cv=None, **kwargs):
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
        if sklearn is None:
            raise ImportError('Please install sklearn before using the `SklearnTuner`.')
        self.scoring = scoring
        if metrics is None:
            metrics = []
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics
        self.cv = cv or sklearn.model_selection.KFold(5, shuffle=True, random_state=1)

    def search(self, X, y, sample_weight=None, groups=None):
        """Performs hyperparameter search.

        Args:
            X: See docstring for `model.fit` for the `sklearn` Models being
                tuned.
            y: See docstring for `model.fit` for the `sklearn` Models being
                tuned.
            sample_weight: Optional. See docstring for `model.fit` for the
                `sklearn` Models being tuned.
            groups: Optional. Required for `sklearn.model_selection` Splitter
                classes that split based on group labels (For example, see
                `sklearn.model_selection.GroupKFold`).
        """
        return super().search(X, y, sample_weight=sample_weight, groups=groups)

    def run_trial(self, trial, X, y, sample_weight=None, groups=None):
        metrics = collections.defaultdict(list)
        cv_kwargs = {'groups': groups} if groups is not None else {}
        for train_indices, test_indices in self.cv.split(X, y, **cv_kwargs):
            X_train = split_data(X, train_indices)
            y_train = split_data(y, train_indices)
            X_test = split_data(X, test_indices)
            y_test = split_data(y, test_indices)
            sample_weight_train = sample_weight[train_indices] if sample_weight is not None else None
            model = self.hypermodel.build(trial.hyperparameters)
            supports_sw = 'sample_weight' in inspect.getfullargspec(model.fit).args
            if isinstance(model, sklearn.pipeline.Pipeline) or not supports_sw:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weight_train)
            sample_weight_test = sample_weight[test_indices] if sample_weight is not None else None
            if self.scoring is None:
                score = model.score(X_test, y_test, sample_weight=sample_weight_test)
            else:
                score = self.scoring(model, X_test, y_test, sample_weight=sample_weight_test)
            metrics['score'].append(score)
            if self.metrics:
                y_test_pred = model.predict(X_test)
                for metric in self.metrics:
                    result = metric(y_test, y_test_pred, sample_weight=sample_weight_test)
                    metrics[metric.__name__].append(result)
        self.save_model(trial.trial_id, model)
        return {name: np.mean(values) for name, values in metrics.items()}

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), 'model.pickle')
        with backend.io.File(fname, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), 'model.pickle')
        with backend.io.File(fname, 'rb') as f:
            return pickle.load(f)