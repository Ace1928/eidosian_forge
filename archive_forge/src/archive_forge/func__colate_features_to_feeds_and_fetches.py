from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
from tensorflow.python.util.all_util import remove_undocumented
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys as _feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as _head
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils as _model_utils
def _colate_features_to_feeds_and_fetches(signature, features, graph, continue_from=None):
    """Uses a saved model signature to construct feed and fetch dictionaries."""
    if continue_from is None:
        state_values = {}
    elif _feature_keys.FilteringResults.STATE_TUPLE in continue_from:
        state_values = _head.state_to_dictionary(continue_from[_feature_keys.FilteringResults.STATE_TUPLE])
    else:
        state_values = continue_from
    input_feed_tensors_by_name = {input_key: graph.as_graph_element(input_value.name) for input_key, input_value in signature.inputs.items()}
    output_tensors_by_name = {output_key: graph.as_graph_element(output_value.name) for output_key, output_value in signature.outputs.items()}
    feed_dict = {}
    for state_key, state_value in state_values.items():
        feed_dict[input_feed_tensors_by_name[state_key]] = state_value
    for feature_key, feature_value in features.items():
        feed_dict[input_feed_tensors_by_name[feature_key]] = feature_value
    return (output_tensors_by_name, feed_dict)