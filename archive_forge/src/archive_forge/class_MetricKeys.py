from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow_estimator.python.estimator import model_fn
class MetricKeys(object):
    """Metric key strings."""
    LOSS = model_fn.LOSS_METRIC_KEY
    LOSS_MEAN = model_fn.AVERAGE_LOSS_METRIC_KEY
    LOSS_REGULARIZATION = 'regularization_loss'
    ACCURACY = 'accuracy'
    PRECISION = 'precision'
    RECALL = 'recall'
    ACCURACY_BASELINE = 'accuracy_baseline'
    AUC = 'auc'
    AUC_PR = 'auc_precision_recall'
    LABEL_MEAN = 'label/mean'
    PREDICTION_MEAN = 'prediction/mean'
    ACCURACY_AT_THRESHOLD = 'accuracy/positive_threshold_%g'
    PRECISION_AT_THRESHOLD = 'precision/positive_threshold_%g'
    RECALL_AT_THRESHOLD = 'recall/positive_threshold_%g'
    PRECISION_AT_RECALL = 'precision_at_recall_%g'
    RECALL_AT_PRECISION = 'recall_at_precision_%g'
    SENSITIVITY_AT_SPECIFICITY = 'sensitivity_at_specificity_%g'
    SPECIFICITY_AT_SENSITIVITY = 'specificity_at_sensitivity_%g'
    PROBABILITY_MEAN_AT_CLASS = 'probability_mean/class%d'
    AUC_AT_CLASS = 'auc/class%d'
    AUC_PR_AT_CLASS = 'auc_precision_recall/class%d'
    PROBABILITY_MEAN_AT_NAME = 'probability_mean/%s'
    AUC_AT_NAME = 'auc/%s'
    AUC_PR_AT_NAME = 'auc_precision_recall/%s'