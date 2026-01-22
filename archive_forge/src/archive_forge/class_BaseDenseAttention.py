from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
class BaseDenseAttention(Layer):
    """Base Attention class for Dense networks.

  This class is suitable for Dense or CNN networks, and not for RNN networks.

  Implementations of attention mechanisms should inherit from this class, and
  reuse the `apply_attention_scores()` method.

  Args:
    causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      attention scores.

  Call Args:

    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
        given, will use `value` for both `key` and `value`, which is the
        most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (no dropout).
    return_attention_scores: bool, it `True`, returns the attention scores
      (after masking and softmax) as an additional output argument.

  Output:

    Attention outputs of shape `[batch_size, Tq, dim]`.
    [Optional] Attention scores after masking and softmax with shape
      `[batch_size, Tq, Tv]`.
  """

    def __init__(self, causal=False, dropout=0.0, **kwargs):
        super(BaseDenseAttention, self).__init__(**kwargs)
        self.causal = causal
        self.dropout = dropout
        self.supports_masking = True

    def _calculate_scores(self, query, key):
        """Calculates attention scores.

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.

    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
        return NotImplementedError

    def _apply_scores(self, scores, value, scores_mask=None, training=None):
        """Applies attention scores to the given value tensor.

    To use this method in your attention layer, follow the steps:

    * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of shape
      `[batch_size, Tv]` to calculate the attention `scores`.
    * Pass `scores` and `value` tensors to this method. The method applies
      `scores_mask`, calculates `attention_distribution = softmax(scores)`, then
      returns `matmul(attention_distribution, value).
    * Apply `query_mask` and return the result.

    Args:
      scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
      value: Value tensor of shape `[batch_size, Tv, dim]`.
      scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
        `[batch_size, Tq, Tv]`. If given, scores at positions where
        `scores_mask==False` do not contribute to the result. It must contain
        at least one `True` value in each line along the last dimension.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (no dropout).

    Returns:
      Tensor of shape `[batch_size, Tq, dim]`.
      Attention scores after masking and softmax with shape
        `[batch_size, Tq, Tv]`.
    """
        if scores_mask is not None:
            padding_mask = math_ops.logical_not(scores_mask)
            if scores.dtype is dtypes.float16:
                scores -= 65504.0 * math_ops.cast(padding_mask, dtype=scores.dtype)
            else:
                scores -= 1000000000.0 * math_ops.cast(padding_mask, dtype=scores.dtype)
        if training is None:
            training = backend.learning_phase()
        weights = nn.softmax(scores)

        def dropped_weights():
            return nn.dropout(weights, rate=self.dropout)
        weights = control_flow_util.smart_cond(training, dropped_weights, lambda: array_ops.identity(weights))
        return (math_ops.matmul(weights, value), weights)

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            v_mask = array_ops.expand_dims(v_mask, axis=-2)
        if self.causal:
            scores_shape = array_ops.shape(scores)
            causal_mask_shape = array_ops.concat([array_ops.ones_like(scores_shape[:-2]), scores_shape[-2:]], axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attention_scores = self._apply_scores(scores=scores, value=v, scores_mask=scores_mask, training=training)
        if q_mask is not None:
            q_mask = array_ops.expand_dims(q_mask, axis=-1)
            result *= math_ops.cast(q_mask, dtype=result.dtype)
        if return_attention_scores:
            return (result, attention_scores)
        return result

    def compute_mask(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        if mask:
            q_mask = mask[0]
            if q_mask is None:
                return None
            return tensor_conversion.convert_to_tensor_v2_with_dispatch(q_mask)
        return None

    def _validate_call_args(self, inputs, mask):
        """Validates arguments of the call method."""
        class_name = self.__class__.__name__
        if not isinstance(inputs, list):
            raise ValueError('{} layer must be called on a list of inputs, namely [query, value] or [query, value, key].'.format(class_name))
        if len(inputs) < 2 or len(inputs) > 3:
            raise ValueError('{} layer accepts inputs list of length 2 or 3, namely [query, value] or [query, value, key]. Given length: {}'.format(class_name, len(inputs)))
        if mask:
            if not isinstance(mask, list):
                raise ValueError('{} layer mask must be a list, namely [query_mask, value_mask].'.format(class_name))
            if len(mask) < 2 or len(mask) > len(inputs):
                raise ValueError('{} layer mask must be a list of length 2, namely [query_mask, value_mask]. Given length: {}'.format(class_name, len(mask)))

    def get_config(self):
        config = {'causal': self.causal, 'dropout': self.dropout}
        base_config = super(BaseDenseAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))