from __future__ import annotations
import math
from typing import Optional, Tuple
import tensorflow as tf
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, invert_attention_mask, stable_softmax
from ...utils import add_start_docstrings_to_model_forward, logging
from .configuration_blip import BlipTextConfig
class TFBlipTextLMHeadModel(TFBlipTextPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ['pooler']
    _keys_to_ignore_on_load_missing = ['position_ids', 'predictions.decoder.bias']

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.bert = TFBlipTextModel(config, add_pooling_layer=False, name='bert')
        self.cls = TFBlipTextOnlyMLMHead(config, name='cls')
        self.label_smoothing = config.label_smoothing

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    @unpack_inputs
    def call(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, past_key_values=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, return_logits=False, is_decoder=True, training=None):
        """
        encoder_hidden_states (`tf.Tensor`, *optional*): Sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model is
            configured as a decoder.
        encoder_attention_mask (`tf.Tensor`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (`tf.Tensor`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(tf.Tensor))`, *optional*):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, is_decoder=is_decoder, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        if return_logits:
            return prediction_scores[:, :-1, :]
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :]
            shifted_prediction_scores = tf.reshape(shifted_prediction_scores, (-1, self.config.vocab_size))
            labels = labels[:, 1:]
            labels = tf.reshape(labels, (-1,))
            one_hot_labels = tf.one_hot(tf.nn.relu(labels), depth=self.config.vocab_size, dtype=tf.float32)
            loss_fct = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=self.label_smoothing, reduction='none')
            masked_positions = tf.cast(tf.not_equal(labels, -100), dtype=tf.float32)
            lm_loss = loss_fct(one_hot_labels, shifted_prediction_scores)
            lm_loss *= masked_positions
            lm_loss = tf.reduce_sum(lm_loss, axis=0) / tf.math.count_nonzero(masked_positions, dtype=tf.float32)
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (lm_loss,) + output if lm_loss is not None else output
        return TFCausalLMOutputWithCrossAttentions(loss=lm_loss, logits=prediction_scores, past_key_values=outputs.past_key_values, hidden_states=outputs.hidden_states, attentions=outputs.attentions, cross_attentions=outputs.cross_attentions)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'past_key_values': past_key_values, 'encoder_hidden_states': model_kwargs.get('encoder_hidden_states', None), 'encoder_attention_mask': model_kwargs.get('encoder_attention_mask', None), 'is_decoder': True}

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past)),)
        return reordered_past

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, 'bert', None) is not None:
            with tf.name_scope(self.bert.name):
                self.bert.build(None)
        if getattr(self, 'cls', None) is not None:
            with tf.name_scope(self.cls.name):
                self.cls.build(None)