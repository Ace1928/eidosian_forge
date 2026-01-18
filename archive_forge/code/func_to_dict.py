import functools
import operator
from ...configuration_utils import PretrainedConfig
from ...utils import logging
def to_dict(self):
    """
        Serializes this instance to a Python dictionary.
        """
    output = super().to_dict()
    output['hidden_dropout'] = output.pop('_hidden_dropout')
    return output