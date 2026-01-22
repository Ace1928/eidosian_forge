from typing import Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils
class SegmentationHead(ClassificationHead):
    """Segmentation layers.

    Use sigmoid and binary crossentropy for binary element segmentation.
    Use softmax and categorical crossentropy for multi-class
    (more than 2) segmentation. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary element segmentation.

    The raw labels will be encoded to 0s and 1s if two classes were found, or
    one-hot encoded if more than two classes were found.
    One pixel only corresponds to one label.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(self, num_classes: Optional[int]=None, loss: Optional[types.LossType]=None, metrics: Optional[types.MetricsType]=None, dropout: Optional[float]=None, **kwargs):
        super().__init__(loss=loss, metrics=metrics, num_classes=num_classes, dropout=dropout, **kwargs)

    def build(self, hp, inputs):
        return inputs

    def get_adapter(self):
        return adapters.SegmentationHeadAdapter(name=self.name)