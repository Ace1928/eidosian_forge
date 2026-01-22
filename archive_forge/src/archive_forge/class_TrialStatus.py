import hashlib
import random
import time
from keras_tuner.src import protos
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import metrics_tracking
from keras_tuner.src.engine import stateful
@keras_tuner_export(['keras_tuner.engine.trial.TrialStatus'])
class TrialStatus:
    RUNNING = 'RUNNING'
    IDLE = 'IDLE'
    INVALID = 'INVALID'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'

    @staticmethod
    def to_proto(status):
        ts = protos.get_proto().TrialStatus
        if status is None:
            return ts.UNKNOWN
        elif status == TrialStatus.RUNNING:
            return ts.RUNNING
        elif status == TrialStatus.IDLE:
            return ts.IDLE
        elif status == TrialStatus.INVALID:
            return ts.INVALID
        elif status == TrialStatus.STOPPED:
            return ts.STOPPED
        elif status == TrialStatus.COMPLETED:
            return ts.COMPLETED
        elif status == TrialStatus.FAILED:
            return ts.FAILED
        else:
            raise ValueError(f'Unknown status {status}')

    @staticmethod
    def from_proto(proto):
        ts = protos.get_proto().TrialStatus
        if proto == ts.UNKNOWN:
            return None
        elif proto == ts.RUNNING:
            return TrialStatus.RUNNING
        elif proto == ts.IDLE:
            return TrialStatus.IDLE
        elif proto == ts.INVALID:
            return TrialStatus.INVALID
        elif proto == ts.STOPPED:
            return TrialStatus.STOPPED
        elif proto == ts.COMPLETED:
            return TrialStatus.COMPLETED
        elif proto == ts.FAILED:
            return TrialStatus.FAILED
        else:
            raise ValueError(f'Unknown status {proto}')