import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine.input_spec import InputSpec
from keras.src.layers.rnn.base_rnn import RNN
from keras.src.utils import conv_utils
from keras.src.utils import generic_utils
from keras.src.utils import tf_utils
N-Dimensional Base class for convolutional-recurrent layers.

    Args:
      rank: Integer, rank of the convolution, e.g. "2" for 2D convolutions.
      cell: A RNN cell instance. A RNN cell is a class that has: - a
        `call(input_at_t, states_at_t)` method, returning `(output_at_t,
        states_at_t_plus_1)`. The call method of the cell can also take the
        optional argument `constants`, see section "Note on passing external
        constants" below. - a `state_size` attribute. This can be a single
        integer (single state) in which case it is the number of channels of the
        recurrent state (which should be the same as the number of channels of
        the cell output). This can also be a list/tuple of integers (one size
        per state).  In this case, the first entry (`state_size[0]`) should be
        the same as the size of the cell output.
      return_sequences: Boolean. Whether to return the last output. in the
        output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
      input_shape: Use this argument to specify the shape of the input when this
        layer is the first one in a model.
    Call arguments:
      inputs: A (2 + `rank`)D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether a
        given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is for use with cells that use dropout.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
      constants: List of constant tensors to be passed to the cell at each
        timestep.
    Input shape:
      (3 + `rank`)D tensor with shape: `(samples, timesteps, channels,
        img_dimensions...)`
      if data_format='channels_first' or shape: `(samples, timesteps,
        img_dimensions..., channels)` if data_format='channels_last'.
    Output shape:
      - If `return_state`: a list of tensors. The first tensor is the output.
        The remaining tensors are the last states,
        each (2 + `rank`)D tensor with shape: `(samples, filters,
          new_img_dimensions...)` if data_format='channels_first'
        or shape: `(samples, new_img_dimensions..., filters)` if
          data_format='channels_last'. img_dimension values might have changed
          due to padding.
      - If `return_sequences`: (3 + `rank`)D tensor with shape: `(samples,
        timesteps, filters, new_img_dimensions...)` if
        data_format='channels_first'
        or shape: `(samples, timesteps, new_img_dimensions..., filters)` if
          data_format='channels_last'.
      - Else, (2 + `rank`)D tensor with shape: `(samples, filters,
        new_img_dimensions...)` if data_format='channels_first'
        or shape: `(samples, new_img_dimensions..., filters)` if
          data_format='channels_last'.
    Masking: This layer supports masking for input data with a variable number
      of timesteps.
    Note on using statefulness in RNNs: You can set RNN layers to be 'stateful',
      which means that the states computed for the samples in one batch will be
      reused as initial states for the samples in the next batch. This assumes a
      one-to-one mapping between samples in different successive batches.
      To enable statefulness: - Specify `stateful=True` in the layer
      constructor.
        - Specify a fixed batch size for your model, by passing
            - If sequential model: `batch_input_shape=(...)` to the first layer
              in your model.
            - If functional model with 1 or more Input layers:
              `batch_shape=(...)` to all the first layers in your model. This is
              the expected shape of your inputs *including the batch size*. It
              should be a tuple of integers, e.g. `(32, 10, 100, 100, 32)`. for
              rank 2 convolution Note that the image dimensions should be
              specified too. - Specify `shuffle=False` when calling fit(). To
              reset the states of your model, call `.reset_states()` on either a
              specific layer, or on your entire model.
    Note on specifying the initial state of RNNs: You can specify the initial
      state of RNN layers symbolically by calling them with the keyword argument
      `initial_state`. The value of `initial_state` should be a tensor or list
      of tensors representing the initial state of the RNN layer. You can
      specify the initial state of RNN layers numerically by calling
      `reset_states` with the keyword argument `states`. The value of `states`
      should be a numpy array or list of numpy arrays representing the initial
      state of the RNN layer.
    Note on passing external constants to RNNs: You can pass "external"
      constants to the cell using the `constants` keyword argument of
      `RNN.__call__` (as well as `RNN.call`) method. This requires that the
      `cell.call` method accepts the same keyword argument `constants`. Such
      constants can be used to condition the cell transformation on additional
      static inputs (not changing over time), a.k.a. an attention mechanism.
    