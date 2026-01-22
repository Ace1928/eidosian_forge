import datetime
import io
import json
import tempfile
import warnings
import zipfile
import ml_dtypes
import numpy as np
from keras.src import backend
from keras.src.backend.common import global_state
from keras.src.layers.layer import Layer
from keras.src.losses.loss import Loss
from keras.src.metrics.metric import Metric
from keras.src.optimizers.optimizer import Optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.trainers.compile_utils import CompileMetrics
from keras.src.utils import file_utils
from keras.src.utils import naming
from keras.src.version import __version__ as keras_version
def _save_model_to_fileobj(model, fileobj, weights_format):
    with ObjectSharingScope():
        serialized_model_dict = serialize_keras_object(model)
    config_json = json.dumps(serialized_model_dict)
    metadata_json = json.dumps({'keras_version': keras_version, 'date_saved': datetime.datetime.now().strftime('%Y-%m-%d@%H:%M:%S')})
    with zipfile.ZipFile(fileobj, 'w') as zf:
        with zf.open(_METADATA_FILENAME, 'w') as f:
            f.write(metadata_json.encode())
        with zf.open(_CONFIG_FILENAME, 'w') as f:
            f.write(config_json.encode())
        if weights_format == 'h5':
            weights_store = H5IOStore(_VARS_FNAME + '.h5', archive=zf, mode='w')
        elif weights_format == 'npz':
            weights_store = NpzIOStore(_VARS_FNAME + '.npz', archive=zf, mode='w')
        else:
            raise ValueError(f"Unknown `weights_format` argument. Expected 'h5' or 'npz'. Received: weights_format={weights_format}")
        asset_store = DiskIOStore(_ASSETS_DIRNAME, archive=zf, mode='w')
        _save_state(model, weights_store=weights_store, assets_store=asset_store, inner_path='', visited_trackables=set())
        weights_store.close()
        asset_store.close()