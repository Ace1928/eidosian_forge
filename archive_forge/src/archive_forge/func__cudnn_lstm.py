import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _cudnn_lstm(inputs, initial_state_h, initial_state_c, kernel, recurrent_kernel, bias, mask, time_major, go_backwards, return_sequences):
    if mask is not None:
        sequence_lengths = _compute_sequence_length_from_mask(mask, time_major)
    else:
        sequence_lengths = None
    if not time_major and sequence_lengths is None:
        inputs = tf.transpose(inputs, perm=(1, 0, 2))
        seq_axis, batch_axis = (0, 1)
    else:
        seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
    init_h = tf.expand_dims(initial_state_h, axis=seq_axis)
    init_c = tf.expand_dims(initial_state_c, axis=seq_axis)
    weights = tf.split(kernel, 4, axis=1)
    weights += tf.split(recurrent_kernel, 4, axis=1)
    full_bias = tf.concat((tf.zeros_like(bias), bias), 0)
    if tf.sysconfig.get_build_info()['is_rocm_build']:
        weights = [weights[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
        full_bias = tf.split(full_bias, 8, axis=0)
        full_bias = [full_bias[x] for x in (0, 1, 3, 2, 4, 5, 7, 6)]
    params = _standardize_cudnn_weights(weights=weights, biases=tf.split(full_bias, 8), shape=tf.constant([-1]), transpose_weights=True)
    if sequence_lengths is not None:
        if go_backwards:
            inputs = tf.reverse_sequence(inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
        outputs, h, c, _, _ = tf.raw_ops.CudnnRNNV3(input=inputs, input_h=init_h, input_c=init_c, params=params, is_training=True, rnn_mode='lstm', sequence_lengths=sequence_lengths, time_major=time_major)
        if go_backwards:
            outputs = tf.reverse_sequence(outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
            outputs = tf.reverse(outputs, axis=[seq_axis])
    else:
        if go_backwards:
            inputs = tf.reverse(inputs, axis=[0])
        outputs, h, c, _ = tf.raw_ops.CudnnRNN(input=inputs, input_h=init_h, input_c=init_c, params=params, is_training=True, rnn_mode='lstm')
    last_output = outputs[-1]
    if not time_major and sequence_lengths is None and return_sequences:
        outputs = tf.transpose(outputs, perm=[1, 0, 2])
    h = tf.squeeze(h, axis=seq_axis)
    c = tf.squeeze(c, axis=seq_axis)
    if sequence_lengths is not None:
        last_output = h
    if not return_sequences:
        outputs = tf.expand_dims(last_output, axis=0 if time_major else 1)
    return (last_output, outputs, [h, c])