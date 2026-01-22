from functools import wraps
import transformers
from packaging import version
def are_fx_features_available():
    return _fx_features_available