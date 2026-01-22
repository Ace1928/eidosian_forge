import collections
import utils
import tensorflow as tf
class _Stack(tf.Module):
    """Stack of pooling and convolutional blocks with residual connections."""

    def __init__(self, num_ch, num_blocks):
        super(_Stack, self).__init__(name='stack')
        self._conv = tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same')
        self._max_pool = tf.keras.layers.MaxPool2D(pool_size=3, padding='same', strides=2)
        self._res_convs0 = [tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_0' % i) for i in range(num_blocks)]
        self._res_convs1 = [tf.keras.layers.Conv2D(num_ch, 3, strides=1, padding='same', name='res_%d/conv2d_1' % i) for i in range(num_blocks)]

    def __call__(self, conv_out):
        conv_out = self._conv(conv_out)
        conv_out = self._max_pool(conv_out)
        for res_conv0, res_conv1 in zip(self._res_convs0, self._res_convs1):
            block_input = conv_out
            conv_out = tf.nn.relu(conv_out)
            conv_out = res_conv0(conv_out)
            conv_out = tf.nn.relu(conv_out)
            conv_out = res_conv1(conv_out)
            conv_out += block_input
        return conv_out