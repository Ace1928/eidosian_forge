import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import models
from keras.src.applications import imagenet_utils
from keras.src.layers import VersionAwareLayers
from keras.src.utils import data_utils
from keras.src.utils import layer_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
def MobileNetV3(stack_fn, last_point_ch, input_shape=None, alpha=1.0, model_type='large', minimalistic=False, include_top=True, weights='imagenet', input_tensor=None, classes=1000, pooling=None, dropout_rate=0.2, classifier_activation='softmax', include_preprocessing=True):
    if not (weights in {'imagenet', None} or tf.io.gfile.exists(weights)):
        raise ValueError(f'The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.  Received weights={weights}')
    if weights == 'imagenet' and include_top and (classes != 1000):
        raise ValueError(f'If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000.  Received classes={classes}')
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(layer_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError('input_tensor: ', input_tensor, f'is not type input_tensor.  Received type(input_tensor)={type(input_tensor)}')
        if is_input_t_tensor:
            if backend.image_data_format() == 'channels_first':
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(f'When backend.image_data_format()=channels_first, input_shape[1] must equal backend.int_shape(input_tensor)[1].  Received input_shape={input_shape}, backend.int_shape(input_tensor)={backend.int_shape(input_tensor)}')
            elif backend.int_shape(input_tensor)[2] != input_shape[1]:
                raise ValueError(f'input_shape[1] must equal backend.int_shape(input_tensor)[2].  Received input_shape={input_shape}, backend.int_shape(input_tensor)={backend.int_shape(input_tensor)}')
        else:
            raise ValueError('input_tensor specified: ', input_tensor, 'is not a keras tensor')
    if input_shape is None and input_tensor is not None:
        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError('input_tensor: ', input_tensor, 'is type: ', type(input_tensor), 'which is not a valid type')
        if backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == 'channels_first':
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
                input_shape = (3, cols, rows)
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]
                input_shape = (cols, rows, 3)
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 3)
    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError(f'Input size must be at least 32x32; Received `input_shape={input_shape}`')
    if weights == 'imagenet':
        if not minimalistic and alpha not in [0.75, 1.0] or (minimalistic and alpha != 1.0):
            raise ValueError('If imagenet weights are being loaded, alpha can be one of `0.75`, `1.0` for non minimalistic or `1.0` for minimalistic only.')
        if rows != cols or rows != 224:
            logging.warning('`input_shape` is undefined or non-square, or `rows` is not 224. Weights for input shape (224, 224) will be loaded as the default.')
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    elif not backend.is_keras_tensor(input_tensor):
        img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
        img_input = input_tensor
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25
    x = img_input
    if include_preprocessing:
        x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)
    x = layers.Conv2D(16, kernel_size=3, strides=(2, 2), padding='same', use_bias=False, name='Conv')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=0.001, momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)
    x = stack_fn(x, kernel, activation, se_ratio)
    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    x = layers.Conv2D(last_conv_ch, kernel_size=1, padding='same', use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis, epsilon=0.001, momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(keepdims=True)(x)
        x = layers.Conv2D(last_point_ch, kernel_size=1, padding='same', use_bias=True, name='Conv_2')(x)
        x = activation(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
        x = layers.Flatten()(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Activation(activation=classifier_activation, name='Predictions')(x)
    elif pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    model = models.Model(inputs, x, name='MobilenetV3' + model_type)
    if weights == 'imagenet':
        model_name = '{}{}_224_{}_float'.format(model_type, '_minimalistic' if minimalistic else '', str(alpha))
        if include_top:
            file_name = 'weights_mobilenet_v3_' + model_name + '.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = 'weights_mobilenet_v3_' + model_name + '_no_top_v2.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(file_name, BASE_WEIGHT_PATH + file_name, cache_subdir='models', file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)
    return model