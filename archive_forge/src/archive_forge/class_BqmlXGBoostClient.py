import logging
from bigquery_ml_utils.inference.xgboost_predictor import Predictor
from google.cloud.ml.prediction import copy_model_to_local
from google.cloud.ml.prediction import ENGINE
from google.cloud.ml.prediction import ENGINE_RUN_TIME
from google.cloud.ml.prediction import FRAMEWORK
from google.cloud.ml.prediction import LOCAL_MODEL_PATH
from google.cloud.ml.prediction import PredictionClient
from google.cloud.ml.prediction import PredictionError
from google.cloud.ml.prediction import SESSION_RUN_TIME
from google.cloud.ml.prediction import Stats
from google.cloud.ml.prediction.frameworks.sk_xg_prediction_lib import SklearnModel
class BqmlXGBoostClient(PredictionClient):
    """The implementation of BQML's XGboost Client."""

    def __init__(self, predictor):
        self._predictor = predictor

    def predict(self, inputs, stats=None, **kwargs):
        stats = stats or Stats()
        stats[FRAMEWORK] = BQML_XGBOOST_FRAMEWORK_NAME
        stats[ENGINE] = BQML_XGBOOST_FRAMEWORK_NAME
        with stats.time(SESSION_RUN_TIME):
            try:
                return self._predictor.predict(inputs, **kwargs)
            except Exception as e:
                logging.exception('Exception during predicting with bqml xgboost model.')
                raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL, 'Exception during predicting with bqml xgboost model: ' + str(e)) from e