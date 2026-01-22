import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _cudnn_gru(inputs, initial_state, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, return_sequences):
    """GRU with cuDNN implementation which is only available for GPU."""
    if mask is not None:
        sequence_lengths = _compute_sequence_length_from_mask(mask, time_major)
    else:
        sequence_lengths = None
    if not time_major and sequence_lengths is None:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        seq_axis, batch_axis = (0, 1)
    else:
        seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
    init_h = tf.expand_dims(initial_state, axis=seq_axis)
    weights = tf.split(kernel, 3, axis=1)
    weights += tf.split(recurrent_kernel, 3, axis=1)
    bias = tf.split(tf.reshape(bias, [-1]), 6)
    if tf.sysconfig.get_build_info()['is_cuda_build']:
        weights[0], weights[1] = (weights[1], weights[0])
        weights[3], weights[4] = (weights[4], weights[3])
        bias[0], bias[1] = (bias[1], bias[0])
        bias[3], bias[4] = (bias[4], bias[3])
    params = _standardize_cudnn_weights(weights=weights, biases=bias, shape=tf.constant([-1]), transpose_weights=True)
    if sequence_lengths is not None:
        if go_backwards:
            inputs = tf.reverse_sequence(inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
        outputs, h, _, _, _ = tf.raw_ops.CudnnRNNV3(input=inputs, input_h=init_h, input_c=0, params=params, is_training=True, rnn_mode='gru', sequence_lengths=sequence_lengths, time_major=time_major)
        if go_backwards:
            outputs = tf.reverse_sequence(outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
            outputs = tf.reverse(outputs, axis=[seq_axis])
    else:
        if go_backwards:
            inputs = tf.reverse(inputs, axis=[0])
        outputs, h, _, _ = tf.raw_ops.CudnnRNN(input=inputs, input_h=init_h, input_c=0, params=params, is_training=True, rnn_mode='gru')
    last_output = outputs[-1]
    if not time_major and sequence_lengths is None and return_sequences:
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
    state = tf.squeeze(h, axis=seq_axis)
    if sequence_lengths is not None:
        last_output = state
    if not return_sequences:
        outputs = tf.expand_dims(last_output, axis=0 if time_major else 1)
    return (last_output, outputs, state)