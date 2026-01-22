import json
import urllib.parse
from tensorboard import context
from tensorboard import errors
def _parse_potential_query_param_flags(self, query_string):
    if not query_string:
        return {}
    try:
        query_string_json = urllib.parse.parse_qs(query_string)
    except ValueError:
        return {}
    potential_feature_flags = query_string_json.get('tensorBoardFeatureFlags', [])
    if not potential_feature_flags:
        return {}
    try:
        client_feature_flags = json.loads(potential_feature_flags[0])
    except json.JSONDecodeError:
        raise errors.InvalidArgumentError('tensorBoardFeatureFlags cannot be JSON decoded.')
    if not isinstance(client_feature_flags, dict):
        raise errors.InvalidArgumentError('tensorBoardFeatureFlags cannot be decoded to a dict.')
    return client_feature_flags