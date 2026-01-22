import base64
import collections
import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import PredictionError
import six
import tensorflow as tf
class TensorFlowModel(prediction_utils.BaseModel):
    """The default implementation of the Model interface that uses TensorFlow.

  This implementation optionally performs preprocessing and postprocessing
  using the provided functions. These functions accept a single instance
  as input and produce a corresponding output to send to the prediction
  client.
  """

    def _get_columns(self, instances, stats, signature):
        """Columnarize the instances, appending input_name, if necessary.

    Instances are the same instances passed to the predict() method. Since
    models with a single input can accept the raw input without the name,
    we create a dict here with that name.

    This list of instances is then converted into a column-oriented format:
    The result is a dictionary mapping input name to a list of values for just
    that input (one entry per row in the original instances list).

    Args:
      instances: the list of instances as provided to the predict() method.
      stats: Stats object for recording timing information.
      signature: SignatureDef for the current request.

    Returns:
      A dictionary mapping input names to their values.

    Raises:
      PredictionError: if an error occurs during prediction.
    """
        with stats.time(prediction_utils.COLUMNARIZE_TIME):
            columns = columnarize(instances)
            for k, v in six.iteritems(columns):
                if k not in signature.inputs.keys():
                    raise PredictionError(PredictionError.INVALID_INPUTS, 'Unexpected tensor name: %s' % k)
                if isinstance(v, list) and len(v) != len(instances):
                    raise PredictionError(PredictionError.INVALID_INPUTS, 'Input %s was missing in at least one input instance.' % k)
        return columns

    def is_single_input(self, signature):
        """Returns True if the graph only has one input tensor."""
        return len(signature.inputs) == 1

    def is_single_string_input(self, signature):
        """Returns True if the graph only has one string input tensor."""
        if self.is_single_input(signature):
            dtype = list(signature.inputs.values())[0].dtype
            return dtype == dtypes.string.as_datatype_enum
        return False

    def get_signature(self, signature_name=None):
        return self._client.get_signature(signature_name)

    def preprocess(self, instances, stats=None, signature_name=None, **kwargs):
        _, signature = self.get_signature(signature_name)
        preprocessed = self._canonicalize_input(instances, signature)
        return self._get_columns(preprocessed, stats, signature)

    def _canonicalize_input(self, instances, signature):
        """Preprocess single-input instances to be dicts if they aren't already."""
        if not self.is_single_input(signature):
            return instances
        tensor_name = list(signature.inputs.keys())[0]
        return canonicalize_single_tensor_input(instances, tensor_name)

    def postprocess(self, predicted_output, original_input=None, stats=None, signature_name=None, **kwargs):
        """Performs the necessary transformations on the prediction results.

    The transformations include rowifying the predicted results, and also
    making sure that each input/output is a dict mapping input/output alias to
    the value for that input/output.

    Args:
      predicted_output: list of instances returned by the predict() method on
        preprocessed instances.
      original_input: List of instances, before any pre-processing was applied.
      stats: Stats object for recording timing information.
      signature_name: the signature name to find out the signature.
      **kwargs: Additional keyword arguments for postprocessing

    Returns:
      A list which is a dict mapping output alias to the output.
    """
        _, signature = self.get_signature(signature_name)
        with stats.time(prediction_utils.ROWIFY_TIME):

            def listify(value):
                if not hasattr(value, 'shape'):
                    return np.asarray([value], dtype=object)
                elif not value.shape:
                    return np.expand_dims(value, axis=0)
                else:
                    return value
            postprocessed_outputs = {alias: listify(val) for alias, val in six.iteritems(predicted_output)}
            postprocessed_outputs = rowify(postprocessed_outputs)
        postprocessed_outputs = list(postprocessed_outputs)
        with stats.time(prediction_utils.ENCODE_TIME):
            try:
                postprocessed_outputs = encode_base64(postprocessed_outputs, signature.outputs)
            except PredictionError as e:
                logging.exception('Encode base64 failed.')
                raise PredictionError(PredictionError.INVALID_OUTPUTS, 'Prediction failed during encoding instances: {0}'.format(e.error_detail))
            except ValueError as e:
                logging.exception('Encode base64 failed.')
                raise PredictionError(PredictionError.INVALID_OUTPUTS, 'Prediction failed during encoding instances: {0}'.format(e))
            except Exception as e:
                logging.exception('Encode base64 failed.')
                raise PredictionError(PredictionError.INVALID_OUTPUTS, 'Prediction failed during encoding instances')
            return postprocessed_outputs

    @classmethod
    def from_client(cls, client, unused_model_path, **unused_kwargs):
        """Creates a TensorFlowModel from a SessionClient and model data files."""
        return cls(client)

    @property
    def signature_map(self):
        return self._client.signature_map