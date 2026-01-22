import collections
import functools
import re
import threading
import warnings
import numpy as np
import wrapt
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import device
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor
from tensorflow.python.ops import session_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.experimental import mixed_precision_global_state
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
class _FetchMapper(object):
    """Definition of the interface provided by fetch mappers.

  Fetch mappers are utility classes used by the _FetchHandler to handle
  arbitrary structures for the `fetch` argument to `Session.run()`.

  The `fetch` argument can be of various shapes: single tensor or op, list of
  fetches, tuple of fetches, namedtuple of fetches, or dict of fetches.  The
  structures can be arbitrarily nested.

  The low level run() API only wants a list of tensor or op names.  The various
  `_FetchMapper` subclasses below take care of handling the different shapes:
  uniquifying the fetches, and constructing results with the original shape.
  """

    def unique_fetches(self):
        """Return the list of unique tensors or ops needed by this fetch mapper.

    Returns:
      A list of tensors or ops.
    """
        raise NotImplementedError('unique_fetches must be implemented by subclasses')

    def build_results(self, values):
        """Build results that match the original shape of the fetch.

    Args:
      values: List of values returned by run(). The values correspond exactly to
        the list tensors or ops returned by unique_fetches().

    Returns:
      A struct of the same shape as the original fetch object handled by
      this fetch mapper.  In the returned struct, the original fetches are
      replaced by their fetched values.
    """
        raise NotImplementedError('build_results must be implemented by subclasses')

    @staticmethod
    def for_fetch(fetch):
        """Creates fetch mapper that handles the structure of `fetch`.

    The default graph must be the one from which we want to fetch values when
    this function is called.

    Args:
      fetch: An arbitrary fetch structure: singleton, list, tuple, namedtuple,
        or dict.

    Returns:
      An instance of a subclass of `_FetchMapper` that handles the shape.
    """
        if fetch is None:
            raise TypeError(f'Argument `fetch` = {fetch} has invalid type "{type(fetch).__name__}". Cannot be None')
        elif isinstance(fetch, (list, tuple)):
            return _ListFetchMapper(fetch)
        elif isinstance(fetch, collections_abc.Mapping):
            return _DictFetchMapper(fetch)
        elif _is_attrs_instance(fetch):
            return _AttrsFetchMapper(fetch)
        else:
            for tensor_type, fetch_fn, _, _ in _REGISTERED_EXPANSIONS:
                if isinstance(fetch, tensor_type):
                    fetches, contraction_fn = fetch_fn(fetch)
                    return _ElementFetchMapper(fetches, contraction_fn)
        raise TypeError(f'Argument `fetch` = {fetch} has invalid type "{type(fetch).__name__}"')