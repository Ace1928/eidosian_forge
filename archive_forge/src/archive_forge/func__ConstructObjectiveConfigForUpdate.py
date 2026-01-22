from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from apitools.base.py import extra_types
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.ai import util as api_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.ai import model_monitoring_jobs_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _ConstructObjectiveConfigForUpdate(self, existing_monitoring_job, feature_thresholds, feature_attribution_thresholds):
    """Construct monitoring objective config.

    Update the feature thresholds for skew/drift detection to all the existing
    deployed models under the job.
    Args:
      existing_monitoring_job: Existing monitoring job.
      feature_thresholds: Dict or None, key: feature_name, value: thresholds.
      feature_attribution_thresholds: Dict or None, key: feature_name, value:
        attribution score thresholds.

    Returns:
      A list of model monitoring objective config.
    """
    prediction_drift_detection = self._ConstructDriftThresholds(feature_thresholds, feature_attribution_thresholds)
    training_prediction_skew_detection = self._ConstructSkewThresholds(feature_thresholds, feature_attribution_thresholds)
    objective_configs = []
    for objective_config in existing_monitoring_job.modelDeploymentMonitoringObjectiveConfigs:
        if objective_config.objectiveConfig.trainingPredictionSkewDetectionConfig:
            if training_prediction_skew_detection.skewThresholds:
                objective_config.objectiveConfig.trainingPredictionSkewDetectionConfig.skewThresholds = training_prediction_skew_detection.skewThresholds
            if training_prediction_skew_detection.attributionScoreSkewThresholds:
                objective_config.objectiveConfig.trainingPredictionSkewDetectionConfig.attributionScoreSkewThresholds = training_prediction_skew_detection.attributionScoreSkewThresholds
        if objective_config.objectiveConfig.predictionDriftDetectionConfig:
            if prediction_drift_detection.driftThresholds:
                objective_config.objectiveConfig.predictionDriftDetectionConfig.driftThresholds = prediction_drift_detection.driftThresholds
            if prediction_drift_detection.attributionScoreDriftThresholds:
                objective_config.objectiveConfig.predictionDriftDetectionConfig.attributionScoreDriftThresholds = prediction_drift_detection.attributionScoreDriftThresholds
        if training_prediction_skew_detection.attributionScoreSkewThresholds or prediction_drift_detection.attributionScoreDriftThresholds:
            objective_config.objectiveConfig.explanationConfig = api_util.GetMessage('ModelMonitoringObjectiveConfigExplanationConfig', self._version)(enableFeatureAttributes=True)
        objective_configs.append(objective_config)
    return objective_configs