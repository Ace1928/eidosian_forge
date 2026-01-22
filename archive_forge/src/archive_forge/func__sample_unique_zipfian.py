from ._internal import NDArrayBase
from ..base import _Null
def _sample_unique_zipfian(range_max=_Null, shape=_Null, out=None, name=None, **kwargs):
    """Draw random samples from an an approximately log-uniform
    or Zipfian distribution without replacement.

    This operation takes a 2-D shape `(batch_size, num_sampled)`,
    and randomly generates *num_sampled* samples from the range of integers [0, range_max)
    for each instance in the batch.

    The elements in each instance are drawn without replacement from the base distribution.
    The base distribution for this operator is an approximately log-uniform or Zipfian distribution:

      P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)

    Additionaly, it also returns the number of trials used to obtain `num_sampled` samples for
    each instance in the batch.

    Example::

       samples, trials = _sample_unique_zipfian(750000, shape=(4, 8192))
       unique(samples[0]) = 8192
       unique(samples[3]) = 8192
       trials[0] = 16435



    Defined in ../src/operator/random/unique_sample_op.cc:L65

    Parameters
    ----------
    range_max : int, required
        The number of possible classes.
    shape : Shape(tuple), optional, default=None
        2-D shape of the output, where shape[0] is the batch size, and shape[1] is the number of candidates to sample for each batch.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)