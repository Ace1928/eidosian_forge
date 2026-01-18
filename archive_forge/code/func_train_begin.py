from mxnet.gluon.contrib.estimator import EpochEnd, TrainBegin, TrainEnd
from mxnet.gluon.nn import HybridSequential
import mlflow
from mlflow.utils.autologging_utils import ExceptionSafeClass, get_autologging_config
def train_begin(self, estimator, *args, **kwargs):
    mlflow.log_param('num_layers', len(estimator.net))
    if estimator.max_epoch is not None:
        mlflow.log_param('epochs', estimator.max_epoch)
    if estimator.max_batch is not None:
        mlflow.log_param('batches', estimator.max_batch)
    mlflow.log_param('optimizer_name', type(estimator.trainer.optimizer).__name__)
    if hasattr(estimator.trainer.optimizer, 'lr'):
        mlflow.log_param('learning_rate', estimator.trainer.optimizer.lr)
    if hasattr(estimator.trainer.optimizer, 'epsilon'):
        mlflow.log_param('epsilon', estimator.trainer.optimizer.epsilon)