import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import ray
from ray import train
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.data.datasource import SimpleTensorFlowDatasource
from ray.data.extensions import TensorArray
from ray.train import Result
from ray.train.tensorflow import TensorflowTrainer, prepare_dataset_shard
def to_tensor_iterator():
    for batch in dataset.iter_tf_batches(batch_size=batch_size, dtypes=tf.float32):
        yield (batch['image'], batch['label'])