import inspect
import itertools
import logging
import warnings
from collections import OrderedDict, defaultdict, deque
from functools import partial
from six import string_types
def _prep_ordered_arg(desired_length, arguments=None):
    """Ensure list of arguments passed to add_ordered_transitions has the proper length.
    Expands the given arguments and apply same condition, callback
    to all transitions if only one has been given.

    Args:
        desired_length (int): The size of the resulting list
        arguments (optional[str, reference or list]): Parameters to be expanded.
    Returns:
        list: Parameter sets with the desired length.
    """
    arguments = listify(arguments) if arguments is not None else [None]
    if len(arguments) != desired_length and len(arguments) != 1:
        raise ValueError('Argument length must be either 1 or the same length as the number of transitions.')
    if len(arguments) == 1:
        return arguments * desired_length
    return arguments