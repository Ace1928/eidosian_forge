import numpy as np
import pandas as pd
from enum import Enum
from .. import CatBoostError
from ..core import metric_description_or_str_to_str
from ..utils import compute_wx_test
class CaseEvaluationResult:
    """
        CaseEvaluationResults stores aggregated statistics for one EvaluationCase and one metric.
    """

    def __init__(self, case, metric_description, eval_step):
        self._case = case
        self._metric_description = metric_description
        self._fold_metric = pd.Series()
        self._fold_metric_iteration = pd.Series()
        self._fold_curves = dict()
        self._eval_step = eval_step

    def _add(self, model, learning_curve):
        if model.get_case() != self._case:
            raise CatBoostError('Model case should be equal to result case')
        fold_id = model.get_fold_id()
        self._fold_curves[fold_id] = learning_curve
        score = max(learning_curve) if self._metric_description.is_max_optimal() else min(learning_curve)
        position = np.argmax(learning_curve) if self._metric_description.is_max_optimal() else np.argmin(learning_curve)
        self._fold_metric.at[fold_id] = score
        self._fold_metric_iteration.at[fold_id] = position

    def __eq__(self, other):
        return np.all(self._fold_metric == other._fold_metric) and np.all(self._fold_metric_iteration == other._fold_metric_iteration) and (self._fold_curves == other._fold_curves)

    def get_case(self):
        """
            ExecutionCases for this result
        """
        return self._case

    def get_fold_ids(self):
        """

        :return: FoldsIds for which this caseResult was calculated
        """
        return self._fold_curves.keys()

    def get_best_metric_for_fold(self, fold):
        """

        :param fold: id of fold to get result
        :return: best metric value, best metric iteration
        """
        return (self._fold_metric[fold], self._fold_metric_iteration[fold])

    def get_best_iterations(self):
        """

        :return: pandas Series with best iterations on all folds
        """
        return self._fold_metric_iteration

    def get_best_metrics(self):
        """

        :return: pandas series with best metric values
        """
        return self._fold_metric

    def get_fold_curve(self, fold):
        """

        :param fold:
        :return: fold learning curve (test scores on every eval_period iteration)
        """
        return self._fold_curves[fold]

    def get_metric_description(self):
        """

        :return: Metric used to build this CaseEvaluationResult
        """
        return self._metric_description

    def get_eval_step(self):
        """

        :return: step which was used for metric computations
        """
        return self._eval_step

    def count_under_and_over_fits(self, overfit_border=0.15, underfit_border=0.95):
        """

        :param overfit_border: min fraction of iterations until overfitting starts one expects all models to have
        :param underfit_border: border, after which there should be no best_metric_scores
        :return: #models with best_metric > underfit_border * iter_count, #models, with best_metric > overfit_border
        """
        count_overfitting = 0
        count_underfitting = 0
        for fold_id, fold_curve in self._fold_curves.items():
            best_score_position = self._fold_metric_iteration[fold_id]
            best_model_size_fraction = best_score_position * 1.0 / len(fold_curve)
            if best_model_size_fraction > overfit_border:
                count_underfitting += 1
            elif best_model_size_fraction < underfit_border:
                count_overfitting += 1
        return (count_overfitting, count_underfitting)

    def estimate_fit_quality(self):
        """

        :return: Simple sanity check that all models overfit and not too fast
        """
        count_overfitting, count_underfitting = self.count_under_and_over_fits()
        if count_overfitting > count_underfitting:
            return 'Overfitting'
        if count_underfitting > count_overfitting:
            return 'Underfitting'
        return 'Good'

    def create_learning_curves_plot(self, offset=None):
        """

        :param offset: First iteration to plot
        :return: plotly Figure with learning curves for each fold
        """
        import plotly.graph_objs as go
        traces = []
        for fold in self.get_fold_ids():
            scores_curve = self.get_fold_curve(fold)
            if offset is not None:
                first_idx = offset
            else:
                first_idx = int(len(scores_curve) * 0.1)
            traces.append(go.Scatter(x=[i * int(self._eval_step) for i in range(first_idx, len(scores_curve))], y=scores_curve[first_idx:], mode='lines', name='Fold #{}'.format(fold)))
        layout = go.Layout(title='Learning curves for case {}'.format(self._case), hovermode='closest', xaxis=dict(title='Iteration', ticklen=5, zeroline=False, gridwidth=2), yaxis=dict(title='Metric', ticklen=5, gridwidth=2), showlegend=True)
        fig = go.Figure(data=traces, layout=layout)
        return fig