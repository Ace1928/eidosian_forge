import sys
from typing import Callable
from typing import Dict
from typing import List
from typing import Union
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import layers
from keras.src.applications import imagenet_utils
from keras.src.engine import training
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def ResNetRS(depth: int, input_shape=None, bn_momentum=0.0, bn_epsilon=1e-05, activation: str='relu', se_ratio=0.25, dropout_rate=0.25, drop_connect_rate=0.2, include_top=True, block_args: List[Dict[str, int]]=None, model_name='resnet-rs', pooling=None, weights='imagenet', input_tensor=None, classes=1000, classifier_activation: Union[str, Callable]='softmax', include_preprocessing=True):
    """Build Resnet-RS model, given provided parameters.

    Args:
        depth: Depth of ResNet network.
        input_shape: optional shape tuple. It should have exactly 3 inputs
          channels, and width and height should be no smaller than 32. E.g.
          (200, 200, 3) would be one valid value.
        bn_momentum: Momentum parameter for Batch Normalization layers.
        bn_epsilon: Epsilon parameter for Batch Normalization layers.
        activation: activation function.
        se_ratio: Squeeze and Excitation layer ratio.
        dropout_rate: dropout rate before final classifier layer.
        drop_connect_rate: dropout rate at skip connections.
        include_top: whether to include the fully-connected layer at the top of
          the network.
        block_args: list of dicts, parameters to construct block modules.
        model_name: name of the model.
        pooling: optional pooling mode for feature extraction when `include_top`
          is `False`.
          - `None` means that the output of the model will be the 4D tensor
            output of the last convolutional layer.
          - `avg` means that global average pooling will be applied to the
            output of the last convolutional layer, and thus the output of the
            model will be a 2D tensor.
          - `max` means that global max pooling will be applied.
        weights: one of `None` (random initialization), `'imagenet'`
          (pre-training on ImageNet), or the path to the weights file to be
          loaded. Note- one model can have multiple imagenet variants depending
          on input shape it was trained with. For input_shape 224x224 pass
          `imagenet-i224` as argument. By default, highest input shape weights
          are downloaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to
          use as image input for the model.
        classes: optional number of classes to classify images into, only to be
          specified if `include_top` is True, and if no `weights` argument is
          specified.
        classifier_activation: A `str` or callable. The activation function to
          use on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.
        include_preprocessing: Boolean, whether to include the preprocessing
          layer (`Rescaling`) at the bottom of the network. Note - Input image
          is normalized by ImageNet mean and standard deviation.
          Defaults to `True`.


    Returns:
        A `tf.keras.Model` instance.

    Raises:
        ValueError: in case of invalid argument for `weights`, or invalid input
            shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
            using a pretrained top layer.
    """
    available_weight_variants = DEPTH_TO_WEIGHT_VARIANTS[depth]
    if weights == 'imagenet':
        max_input_shape = max(available_weight_variants)
        weights = f'{weights}-i{max_input_shape}'
    weights_allow_list = [f'imagenet-i{x}' for x in available_weight_variants]
    if not (weights in {*weights_allow_list, None} or tf.io.gfile.exists(weights)):
        raise ValueError(f"The `weights` argument should be either `None` (random initialization), `'imagenet'` (pre-training on ImageNet, with highest available input shape), or the path to the weights file to be loaded. For ResNetRS{depth} the following weight variants are available {weights_allow_list} (default=highest). Received weights={weights}")
    if weights in weights_allow_list and include_top and (classes != 1000):
        raise ValueError(f"If using `weights` as `'imagenet'` or any of {weights_allow_list} with `include_top` as true, `classes` should be 1000. Received classes={classes}")
    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=include_top, weights=weights)
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    x = img_input
    if include_preprocessing:
        num_channels = input_shape[bn_axis - 1]
        x = layers.Rescaling(scale=1.0 / 255)(x)
        if num_channels == 3:
            x = layers.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2], axis=bn_axis)(x)
    x = STEM(bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, activation=activation)(x)
    if block_args is None:
        block_args = BLOCK_ARGS[depth]
    for i, args in enumerate(block_args):
        survival_probability = get_survival_probability(init_rate=drop_connect_rate, block_num=i + 2, total_blocks=len(block_args) + 1)
        x = BlockGroup(filters=args['input_filters'], activation=activation, strides=1 if i == 0 else 2, num_repeats=args['num_repeats'], se_ratio=se_ratio, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, survival_probability=survival_probability, name=f'BlockGroup{i + 2}_')(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = training.Model(inputs, x, name=model_name)
    if weights in weights_allow_list:
        weights_input_shape = weights.split('-')[-1]
        weights_name = f'{model_name}-{weights_input_shape}'
        if not include_top:
            weights_name += '_notop'
        filename = f'{weights_name}.h5'
        download_url = BASE_WEIGHTS_URL + filename
        weights_path = data_utils.get_file(fname=filename, origin=download_url, cache_subdir='models', file_hash=WEIGHT_HASHES[filename])
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model