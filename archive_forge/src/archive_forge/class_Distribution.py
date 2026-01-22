import abc
import contextlib
import types
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['distributions.Distribution'])
class Distribution(_BaseDistribution, metaclass=_DistributionMeta):
    """A generic probability distribution base class.

  `Distribution` is a base class for constructing and organizing properties
  (e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian).

  #### Subclassing

  Subclasses are expected to implement a leading-underscore version of the
  same-named function. The argument signature should be identical except for
  the omission of `name="..."`. For example, to enable `log_prob(value,
  name="log_prob")` a subclass should implement `_log_prob(value)`.

  Subclasses can append to public-level docstrings by providing
  docstrings for their method specializations. For example:

  ```python
  @util.AppendDocstring("Some other details.")
  def _log_prob(self, value):
    ...
  ```

  would add the string "Some other details." to the `log_prob` function
  docstring. This is implemented as a simple decorator to avoid python
  linter complaining about missing Args/Returns/Raises sections in the
  partial docstrings.

  #### Broadcasting, batching, and shapes

  All distributions support batches of independent distributions of that type.
  The batch shape is determined by broadcasting together the parameters.

  The shape of arguments to `__init__`, `cdf`, `log_cdf`, `prob`, and
  `log_prob` reflect this broadcasting, as does the return value of `sample` and
  `sample_n`.

  `sample_n_shape = [n] + batch_shape + event_shape`, where `sample_n_shape` is
  the shape of the `Tensor` returned from `sample_n`, `n` is the number of
  samples, `batch_shape` defines how many independent distributions there are,
  and `event_shape` defines the shape of samples from each of those independent
  distributions. Samples are independent along the `batch_shape` dimensions, but
  not necessarily so along the `event_shape` dimensions (depending on the
  particulars of the underlying distribution).

  Using the `Uniform` distribution as an example:

  ```python
  minval = 3.0
  maxval = [[4.0, 6.0],
            [10.0, 12.0]]

  # Broadcasting:
  # This instance represents 4 Uniform distributions. Each has a lower bound at
  # 3.0 as the `minval` parameter was broadcasted to match `maxval`'s shape.
  u = Uniform(minval, maxval)

  # `event_shape` is `TensorShape([])`.
  event_shape = u.event_shape
  # `event_shape_t` is a `Tensor` which will evaluate to [].
  event_shape_t = u.event_shape_tensor()

  # Sampling returns a sample per distribution. `samples` has shape
  # [5, 2, 2], which is [n] + batch_shape + event_shape, where n=5,
  # batch_shape=[2, 2], and event_shape=[].
  samples = u.sample_n(5)

  # The broadcasting holds across methods. Here we use `cdf` as an example. The
  # same holds for `log_cdf` and the likelihood functions.

  # `cum_prob` has shape [2, 2] as the `value` argument was broadcasted to the
  # shape of the `Uniform` instance.
  cum_prob_broadcast = u.cdf(4.0)

  # `cum_prob`'s shape is [2, 2], one per distribution. No broadcasting
  # occurred.
  cum_prob_per_dist = u.cdf([[4.0, 5.0],
                             [6.0, 7.0]])

  # INVALID as the `value` argument is not broadcastable to the distribution's
  # shape.
  cum_prob_invalid = u.cdf([4.0, 5.0, 6.0])
  ```

  #### Shapes

  There are three important concepts associated with TensorFlow Distributions
  shapes:
  - Event shape describes the shape of a single draw from the distribution;
    it may be dependent across dimensions. For scalar distributions, the event
    shape is `[]`. For a 5-dimensional MultivariateNormal, the event shape is
    `[5]`.
  - Batch shape describes independent, not identically distributed draws, aka a
    "collection" or "bunch" of distributions.
  - Sample shape describes independent, identically distributed draws of batches
    from the distribution family.

  The event shape and the batch shape are properties of a Distribution object,
  whereas the sample shape is associated with a specific call to `sample` or
  `log_prob`.

  For detailed usage examples of TensorFlow Distributions shapes, see
  [this tutorial](
  https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Understanding_TensorFlow_Distributions_Shapes.ipynb)

  #### Parameter values leading to undefined statistics or distributions.

  Some distributions do not have well-defined statistics for all initialization
  parameter values. For example, the beta distribution is parameterized by
  positive real numbers `concentration1` and `concentration0`, and does not have
  well-defined mode if `concentration1 < 1` or `concentration0 < 1`.

  The user is given the option of raising an exception or returning `NaN`.

  ```python
  a = tf.exp(tf.matmul(logits, weights_a))
  b = tf.exp(tf.matmul(logits, weights_b))

  # Will raise exception if ANY batch member has a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=False)
  mode = dist.mode().eval()

  # Will return NaN for batch members with either a < 1 or b < 1.
  dist = distributions.beta(a, b, allow_nan_stats=True)  # Default behavior
  mode = dist.mode().eval()
  ```

  In all cases, an exception is raised if *invalid* parameters are passed, e.g.

  ```python
  # Will raise an exception if any Op is run.
  negative_a = -1.0 * a  # beta distribution by definition has a > 0.
  dist = distributions.beta(negative_a, b, allow_nan_stats=True)
  dist.mean().eval()
  ```

  """

    @deprecation.deprecated('2019-01-01', 'The TensorFlow Distributions library has moved to TensorFlow Probability (https://github.com/tensorflow/probability). You should update all references to use `tfp.distributions` instead of `tf.distributions`.', warn_once=True)
    def __init__(self, dtype, reparameterization_type, validate_args, allow_nan_stats, parameters=None, graph_parents=None, name=None):
        """Constructs the `Distribution`.

    **This is a private method for subclass use.**

    Args:
      dtype: The type of the event samples. `None` implies no type-enforcement.
      reparameterization_type: Instance of `ReparameterizationType`.
        If `distributions.FULLY_REPARAMETERIZED`, this
        `Distribution` can be reparameterized in terms of some standard
        distribution with a function whose Jacobian is constant for the support
        of the standard distribution. If `distributions.NOT_REPARAMETERIZED`,
        then no such reparameterization is available.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      parameters: Python `dict` of parameters used to instantiate this
        `Distribution`.
      graph_parents: Python `list` of graph prerequisites of this
        `Distribution`.
      name: Python `str` name prefixed to Ops created by this class. Default:
        subclass name.

    Raises:
      ValueError: if any member of graph_parents is `None` or not a `Tensor`.
    """
        graph_parents = [] if graph_parents is None else graph_parents
        for i, t in enumerate(graph_parents):
            if t is None or not tensor_util.is_tf_type(t):
                raise ValueError('Graph parent item %d is not a Tensor; %s.' % (i, t))
        if not name or name[-1] != '/':
            non_unique_name = name or type(self).__name__
            with ops.name_scope(non_unique_name) as name:
                pass
        self._dtype = dtype
        self._reparameterization_type = reparameterization_type
        self._allow_nan_stats = allow_nan_stats
        self._validate_args = validate_args
        self._parameters = parameters or {}
        self._graph_parents = graph_parents
        self._name = name

    @property
    def _parameters(self):
        return self._parameter_dict

    @_parameters.setter
    def _parameters(self, value):
        """Intercept assignments to self._parameters to avoid reference cycles.

    Parameters are often created using locals(), so we need to clean out any
    references to `self` before assigning it to an attribute.

    Args:
      value: A dictionary of parameters to assign to the `_parameters` property.
    """
        if 'self' in value:
            del value['self']
        self._parameter_dict = value

    @classmethod
    def param_shapes(cls, sample_shape, name='DistributionParamShapes'):
        """Shapes of parameters given the desired shape of a call to `sample()`.

    This is a class method that describes what key/value arguments are required
    to instantiate the given `Distribution` so that a particular shape is
    returned for that instance's call to `sample()`.

    Subclasses should override class method `_param_shapes`.

    Args:
      sample_shape: `Tensor` or python list/tuple. Desired shape of a call to
        `sample()`.
      name: name to prepend ops with.

    Returns:
      `dict` of parameter name to `Tensor` shapes.
    """
        with ops.name_scope(name, values=[sample_shape]):
            return cls._param_shapes(sample_shape)

    @classmethod
    def param_static_shapes(cls, sample_shape):
        """param_shapes with static (i.e. `TensorShape`) shapes.

    This is a class method that describes what key/value arguments are required
    to instantiate the given `Distribution` so that a particular shape is
    returned for that instance's call to `sample()`. Assumes that the sample's
    shape is known statically.

    Subclasses should override class method `_param_shapes` to return
    constant-valued tensors when constant values are fed.

    Args:
      sample_shape: `TensorShape` or python list/tuple. Desired shape of a call
        to `sample()`.

    Returns:
      `dict` of parameter name to `TensorShape`.

    Raises:
      ValueError: if `sample_shape` is a `TensorShape` and is not fully defined.
    """
        if isinstance(sample_shape, tensor_shape.TensorShape):
            if not sample_shape.is_fully_defined():
                raise ValueError('TensorShape sample_shape must be fully defined')
            sample_shape = sample_shape.as_list()
        params = cls.param_shapes(sample_shape)
        static_params = {}
        for name, shape in params.items():
            static_shape = tensor_util.constant_value(shape)
            if static_shape is None:
                raise ValueError('sample_shape must be a fully-defined TensorShape or list/tuple')
            static_params[name] = tensor_shape.TensorShape(static_shape)
        return static_params

    @staticmethod
    def _param_shapes(sample_shape):
        raise NotImplementedError('_param_shapes not implemented')

    @property
    def name(self):
        """Name prepended to all ops created by this `Distribution`."""
        return self._name

    @property
    def dtype(self):
        """The `DType` of `Tensor`s handled by this `Distribution`."""
        return self._dtype

    @property
    def parameters(self):
        """Dictionary of parameters used to instantiate this `Distribution`."""
        return {k: v for k, v in self._parameters.items() if not k.startswith('__') and k != 'self'}

    @property
    def reparameterization_type(self):
        """Describes how samples from the distribution are reparameterized.

    Currently this is one of the static instances
    `distributions.FULLY_REPARAMETERIZED`
    or `distributions.NOT_REPARAMETERIZED`.

    Returns:
      An instance of `ReparameterizationType`.
    """
        return self._reparameterization_type

    @property
    def allow_nan_stats(self):
        """Python `bool` describing behavior when a stat is undefined.

    Stats return +/- infinity when it makes sense. E.g., the variance of a
    Cauchy distribution is infinity. However, sometimes the statistic is
    undefined, e.g., if a distribution's pdf does not achieve a maximum within
    the support of the distribution, the mode is undefined. If the mean is
    undefined, then by definition the variance is undefined. E.g. the mean for
    Student's T for df = 1 is undefined (no clear way to say it is either + or -
    infinity), so the variance = E[(X - mean)**2] is also undefined.

    Returns:
      allow_nan_stats: Python `bool`.
    """
        return self._allow_nan_stats

    @property
    def validate_args(self):
        """Python `bool` indicating possibly expensive checks are enabled."""
        return self._validate_args

    def copy(self, **override_parameters_kwargs):
        """Creates a deep copy of the distribution.

    Note: the copy distribution may continue to depend on the original
    initialization arguments.

    Args:
      **override_parameters_kwargs: String/value dictionary of initialization
        arguments to override with new values.

    Returns:
      distribution: A new instance of `type(self)` initialized from the union
        of self.parameters and override_parameters_kwargs, i.e.,
        `dict(self.parameters, **override_parameters_kwargs)`.
    """
        parameters = dict(self.parameters, **override_parameters_kwargs)
        return type(self)(**parameters)

    def _batch_shape_tensor(self):
        raise NotImplementedError('batch_shape_tensor is not implemented: {}'.format(type(self).__name__))

    def batch_shape_tensor(self, name='batch_shape_tensor'):
        """Shape of a single sample from a single event index as a 1-D `Tensor`.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Args:
      name: name to give to the op

    Returns:
      batch_shape: `Tensor`.
    """
        with self._name_scope(name):
            if self.batch_shape.is_fully_defined():
                return ops.convert_to_tensor(self.batch_shape.as_list(), dtype=dtypes.int32, name='batch_shape')
            return self._batch_shape_tensor()

    def _batch_shape(self):
        return tensor_shape.TensorShape(None)

    @property
    def batch_shape(self):
        """Shape of a single sample from a single event index as a `TensorShape`.

    May be partially defined or unknown.

    The batch dimensions are indexes into independent, non-identical
    parameterizations of this distribution.

    Returns:
      batch_shape: `TensorShape`, possibly unknown.
    """
        return tensor_shape.as_shape(self._batch_shape())

    def _event_shape_tensor(self):
        raise NotImplementedError('event_shape_tensor is not implemented: {}'.format(type(self).__name__))

    def event_shape_tensor(self, name='event_shape_tensor'):
        """Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op

    Returns:
      event_shape: `Tensor`.
    """
        with self._name_scope(name):
            if self.event_shape.is_fully_defined():
                return ops.convert_to_tensor(self.event_shape.as_list(), dtype=dtypes.int32, name='event_shape')
            return self._event_shape_tensor()

    def _event_shape(self):
        return tensor_shape.TensorShape(None)

    @property
    def event_shape(self):
        """Shape of a single sample from a single batch as a `TensorShape`.

    May be partially defined or unknown.

    Returns:
      event_shape: `TensorShape`, possibly unknown.
    """
        return tensor_shape.as_shape(self._event_shape())

    def is_scalar_event(self, name='is_scalar_event'):
        """Indicates that `event_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_event: `bool` scalar `Tensor`.
    """
        with self._name_scope(name):
            return ops.convert_to_tensor(self._is_scalar_helper(self.event_shape, self.event_shape_tensor), name='is_scalar_event')

    def is_scalar_batch(self, name='is_scalar_batch'):
        """Indicates that `batch_shape == []`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      is_scalar_batch: `bool` scalar `Tensor`.
    """
        with self._name_scope(name):
            return ops.convert_to_tensor(self._is_scalar_helper(self.batch_shape, self.batch_shape_tensor), name='is_scalar_batch')

    def _sample_n(self, n, seed=None):
        raise NotImplementedError('sample_n is not implemented: {}'.format(type(self).__name__))

    def _call_sample_n(self, sample_shape, seed, name, **kwargs):
        with self._name_scope(name, values=[sample_shape]):
            sample_shape = ops.convert_to_tensor(sample_shape, dtype=dtypes.int32, name='sample_shape')
            sample_shape, n = self._expand_sample_shape_to_vector(sample_shape, 'sample_shape')
            samples = self._sample_n(n, seed, **kwargs)
            batch_event_shape = array_ops.shape(samples)[1:]
            final_shape = array_ops.concat([sample_shape, batch_event_shape], 0)
            samples = array_ops.reshape(samples, final_shape)
            samples = self._set_sample_static_shape(samples, sample_shape)
            return samples

    def sample(self, sample_shape=(), seed=None, name='sample'):
        """Generate samples of the specified shape.

    Note that a call to `sample()` without arguments will generate a single
    sample.

    Args:
      sample_shape: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
      seed: Python integer seed for RNG
      name: name to give to the op.

    Returns:
      samples: a `Tensor` with prepended dimensions `sample_shape`.
    """
        return self._call_sample_n(sample_shape, seed, name)

    def _log_prob(self, value):
        raise NotImplementedError('log_prob is not implemented: {}'.format(type(self).__name__))

    def _call_log_prob(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._log_prob(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return math_ops.log(self._prob(value, **kwargs))
                except NotImplementedError:
                    raise original_exception

    def log_prob(self, value, name='log_prob'):
        """Log probability density/mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      log_prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
        return self._call_log_prob(value, name)

    def _prob(self, value):
        raise NotImplementedError('prob is not implemented: {}'.format(type(self).__name__))

    def _call_prob(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._prob(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return math_ops.exp(self._log_prob(value, **kwargs))
                except NotImplementedError:
                    raise original_exception

    def prob(self, value, name='prob'):
        """Probability density/mass function.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      prob: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
        return self._call_prob(value, name)

    def _log_cdf(self, value):
        raise NotImplementedError('log_cdf is not implemented: {}'.format(type(self).__name__))

    def _call_log_cdf(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._log_cdf(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return math_ops.log(self._cdf(value, **kwargs))
                except NotImplementedError:
                    raise original_exception

    def log_cdf(self, value, name='log_cdf'):
        """Log cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    log_cdf(x) := Log[ P[X <= x] ]
    ```

    Often, a numerical approximation can be used for `log_cdf(x)` that yields
    a more accurate answer than simply taking the logarithm of the `cdf` when
    `x << -1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      logcdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
        return self._call_log_cdf(value, name)

    def _cdf(self, value):
        raise NotImplementedError('cdf is not implemented: {}'.format(type(self).__name__))

    def _call_cdf(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._cdf(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return math_ops.exp(self._log_cdf(value, **kwargs))
                except NotImplementedError:
                    raise original_exception

    def cdf(self, value, name='cdf'):
        """Cumulative distribution function.

    Given random variable `X`, the cumulative distribution function `cdf` is:

    ```none
    cdf(x) := P[X <= x]
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      cdf: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
        return self._call_cdf(value, name)

    def _log_survival_function(self, value):
        raise NotImplementedError('log_survival_function is not implemented: {}'.format(type(self).__name__))

    def _call_log_survival_function(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._log_survival_function(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return math_ops.log1p(-self.cdf(value, **kwargs))
                except NotImplementedError:
                    raise original_exception

    def log_survival_function(self, value, name='log_survival_function'):
        """Log survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    log_survival_function(x) = Log[ P[X > x] ]
                             = Log[ 1 - P[X <= x] ]
                             = Log[ 1 - cdf(x) ]
    ```

    Typically, different numerical approximations can be used for the log
    survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
        return self._call_log_survival_function(value, name)

    def _survival_function(self, value):
        raise NotImplementedError('survival_function is not implemented: {}'.format(type(self).__name__))

    def _call_survival_function(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            try:
                return self._survival_function(value, **kwargs)
            except NotImplementedError as original_exception:
                try:
                    return 1.0 - self.cdf(value, **kwargs)
                except NotImplementedError:
                    raise original_exception

    def survival_function(self, value, name='survival_function'):
        """Survival function.

    Given random variable `X`, the survival function is defined:

    ```none
    survival_function(x) = P[X > x]
                         = 1 - P[X <= x]
                         = 1 - cdf(x).
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
        `self.dtype`.
    """
        return self._call_survival_function(value, name)

    def _entropy(self):
        raise NotImplementedError('entropy is not implemented: {}'.format(type(self).__name__))

    def entropy(self, name='entropy'):
        """Shannon entropy in nats."""
        with self._name_scope(name):
            return self._entropy()

    def _mean(self):
        raise NotImplementedError('mean is not implemented: {}'.format(type(self).__name__))

    def mean(self, name='mean'):
        """Mean."""
        with self._name_scope(name):
            return self._mean()

    def _quantile(self, value):
        raise NotImplementedError('quantile is not implemented: {}'.format(type(self).__name__))

    def _call_quantile(self, value, name, **kwargs):
        with self._name_scope(name, values=[value]):
            value = _convert_to_tensor(value, name='value', preferred_dtype=self.dtype)
            return self._quantile(value, **kwargs)

    def quantile(self, value, name='quantile'):
        """Quantile function. Aka "inverse cdf" or "percent point function".

    Given random variable `X` and `p in [0, 1]`, the `quantile` is:

    ```none
    quantile(p) := x such that P[X <= x] == p
    ```

    Args:
      value: `float` or `double` `Tensor`.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      quantile: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
        values of type `self.dtype`.
    """
        return self._call_quantile(value, name)

    def _variance(self):
        raise NotImplementedError('variance is not implemented: {}'.format(type(self).__name__))

    def variance(self, name='variance'):
        """Variance.

    Variance is defined as,

    ```none
    Var = E[(X - E[X])**2]
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `Var.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      variance: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    """
        with self._name_scope(name):
            try:
                return self._variance()
            except NotImplementedError as original_exception:
                try:
                    return math_ops.square(self._stddev())
                except NotImplementedError:
                    raise original_exception

    def _stddev(self):
        raise NotImplementedError('stddev is not implemented: {}'.format(type(self).__name__))

    def stddev(self, name='stddev'):
        """Standard deviation.

    Standard deviation is defined as,

    ```none
    stddev = E[(X - E[X])**2]**0.5
    ```

    where `X` is the random variable associated with this distribution, `E`
    denotes expectation, and `stddev.shape = batch_shape + event_shape`.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      stddev: Floating-point `Tensor` with shape identical to
        `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.
    """
        with self._name_scope(name):
            try:
                return self._stddev()
            except NotImplementedError as original_exception:
                try:
                    return math_ops.sqrt(self._variance())
                except NotImplementedError:
                    raise original_exception

    def _covariance(self):
        raise NotImplementedError('covariance is not implemented: {}'.format(type(self).__name__))

    def covariance(self, name='covariance'):
        """Covariance.

    Covariance is (possibly) defined only for non-scalar-event distributions.

    For example, for a length-`k`, vector-valued distribution, it is calculated
    as,

    ```none
    Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
    ```

    where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
    denotes expectation.

    Alternatively, for non-vector, multivariate distributions (e.g.,
    matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
    under some vectorization of the events, i.e.,

    ```none
    Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
    ```

    where `Cov` is a (batch of) `k' x k'` matrices,
    `0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
    mapping indices of this distribution's event dimensions to indices of a
    length-`k'` vector.

    Args:
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      covariance: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
        where the first `n` dimensions are batch coordinates and
        `k' = reduce_prod(self.event_shape)`.
    """
        with self._name_scope(name):
            return self._covariance()

    def _mode(self):
        raise NotImplementedError('mode is not implemented: {}'.format(type(self).__name__))

    def mode(self, name='mode'):
        """Mode."""
        with self._name_scope(name):
            return self._mode()

    def _cross_entropy(self, other):
        return kullback_leibler.cross_entropy(self, other, allow_nan_stats=self.allow_nan_stats)

    def cross_entropy(self, other, name='cross_entropy'):
        """Computes the (Shannon) cross entropy.

    Denote this distribution (`self`) by `P` and the `other` distribution by
    `Q`. Assuming `P, Q` are absolutely continuous with respect to
    one another and permit densities `p(x) dr(x)` and `q(x) dr(x)`, (Shanon)
    cross entropy is defined as:

    ```none
    H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
    ```

    where `F` denotes the support of the random variable `X ~ P`.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      cross_entropy: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of (Shanon) cross entropy.
    """
        with self._name_scope(name):
            return self._cross_entropy(other)

    def _kl_divergence(self, other):
        return kullback_leibler.kl_divergence(self, other, allow_nan_stats=self.allow_nan_stats)

    def kl_divergence(self, other, name='kl_divergence'):
        """Computes the Kullback--Leibler divergence.

    Denote this distribution (`self`) by `p` and the `other` distribution by
    `q`. Assuming `p, q` are absolutely continuous with respect to reference
    measure `r`, the KL divergence is defined as:

    ```none
    KL[p, q] = E_p[log(p(X)/q(X))]
             = -int_F p(x) log q(x) dr(x) + int_F p(x) log p(x) dr(x)
             = H[p, q] - H[p]
    ```

    where `F` denotes the support of the random variable `X ~ p`, `H[., .]`
    denotes (Shanon) cross entropy, and `H[.]` denotes (Shanon) entropy.

    Args:
      other: `tfp.distributions.Distribution` instance.
      name: Python `str` prepended to names of ops created by this function.

    Returns:
      kl_divergence: `self.dtype` `Tensor` with shape `[B1, ..., Bn]`
        representing `n` different calculations of the Kullback-Leibler
        divergence.
    """
        with self._name_scope(name):
            return self._kl_divergence(other)

    def __str__(self):
        return 'tfp.distributions.{type_name}("{self_name}"{maybe_batch_shape}{maybe_event_shape}, dtype={dtype})'.format(type_name=type(self).__name__, self_name=self.name, maybe_batch_shape=', batch_shape={}'.format(self.batch_shape) if self.batch_shape.ndims is not None else '', maybe_event_shape=', event_shape={}'.format(self.event_shape) if self.event_shape.ndims is not None else '', dtype=self.dtype.name)

    def __repr__(self):
        return "<tfp.distributions.{type_name} '{self_name}' batch_shape={batch_shape} event_shape={event_shape} dtype={dtype}>".format(type_name=type(self).__name__, self_name=self.name, batch_shape=self.batch_shape, event_shape=self.event_shape, dtype=self.dtype.name)

    @contextlib.contextmanager
    def _name_scope(self, name=None, values=None):
        """Helper function to standardize op scope."""
        with ops.name_scope(self.name):
            with ops.name_scope(name, values=([] if values is None else values) + self._graph_parents) as scope:
                yield scope

    def _expand_sample_shape_to_vector(self, x, name):
        """Helper to `sample` which ensures input is 1D."""
        x_static_val = tensor_util.constant_value(x)
        if x_static_val is None:
            prod = math_ops.reduce_prod(x)
        else:
            prod = np.prod(x_static_val, dtype=x.dtype.as_numpy_dtype())
        ndims = x.get_shape().ndims
        if ndims is None:
            ndims = array_ops.rank(x)
            expanded_shape = util.pick_vector(math_ops.equal(ndims, 0), np.array([1], dtype=np.int32), array_ops.shape(x))
            x = array_ops.reshape(x, expanded_shape)
        elif ndims == 0:
            if x_static_val is not None:
                x = ops.convert_to_tensor(np.array([x_static_val], dtype=x.dtype.as_numpy_dtype()), name=name)
            else:
                x = array_ops.reshape(x, [1])
        elif ndims != 1:
            raise ValueError('Input is neither scalar nor vector.')
        return (x, prod)

    def _set_sample_static_shape(self, x, sample_shape):
        """Helper to `sample`; sets static shape info."""
        sample_shape = tensor_shape.TensorShape(tensor_util.constant_value(sample_shape))
        ndims = x.get_shape().ndims
        sample_ndims = sample_shape.ndims
        batch_ndims = self.batch_shape.ndims
        event_ndims = self.event_shape.ndims
        if ndims is None and sample_ndims is not None and (batch_ndims is not None) and (event_ndims is not None):
            ndims = sample_ndims + batch_ndims + event_ndims
            x.set_shape([None] * ndims)
        if ndims is not None and sample_ndims is not None:
            shape = sample_shape.concatenate([None] * (ndims - sample_ndims))
            x.set_shape(x.get_shape().merge_with(shape))
        if ndims is not None and event_ndims is not None:
            shape = tensor_shape.TensorShape([None] * (ndims - event_ndims)).concatenate(self.event_shape)
            x.set_shape(x.get_shape().merge_with(shape))
        if batch_ndims is not None:
            if ndims is not None:
                if sample_ndims is None and event_ndims is not None:
                    sample_ndims = ndims - batch_ndims - event_ndims
                elif event_ndims is None and sample_ndims is not None:
                    event_ndims = ndims - batch_ndims - sample_ndims
            if sample_ndims is not None and event_ndims is not None:
                shape = tensor_shape.TensorShape([None] * sample_ndims).concatenate(self.batch_shape).concatenate([None] * event_ndims)
                x.set_shape(x.get_shape().merge_with(shape))
        return x

    def _is_scalar_helper(self, static_shape, dynamic_shape_fn):
        """Implementation for `is_scalar_batch` and `is_scalar_event`."""
        if static_shape.ndims is not None:
            return static_shape.ndims == 0
        shape = dynamic_shape_fn()
        if shape.get_shape().ndims is not None and shape.get_shape().dims[0].value is not None:
            return shape.get_shape().as_list() == [0]
        return math_ops.equal(array_ops.shape(shape)[0], 0)