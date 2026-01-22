from tensorflow.python.eager import backprop
from tensorflow.python.training import optimizer as optimizer_lib
class StandardSingleLossStep(StandardInputStep):
    """A step function that implements a training step for a feed forward network.

  An instance of this class is intended to be used as a callable:

  ```python
  ...
  step = step_fn.StandardSingleLossStep(
      dataset, loss_fn, optimizer, distribution)

  # Run a single training step on a given DistributionStrategy:
  step(distribution)
  ...
  ```

  Args:
    dataset_fn: a function that returns a tf.data Dataset that produces the
      input for the model.
    loss_fn: a function that takes a context and inputs as arguments. It returns
      the loss for those inputs. `context` is an instance of
      `values.MultiStepContext` that will be passed when `loss_fn` is run.
      `context` can be used to specify the outputs to be returned from
      `loss_fn`, among other things.
    optimizer: an optimizer that implements an update rule.
    distribution: a `DistributionStrategy` object.
  """

    def __init__(self, dataset_fn, loss_fn, optimizer, distribution, iterations_per_step=1):
        super(StandardSingleLossStep, self).__init__(dataset_fn, distribution)
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._iterations_per_step = iterations_per_step

    def __call__(self):
        with self._distribution.scope():

            def step_fn(ctx, inputs):
                """Function to run one iteration with one input."""
                gradients_fn = backprop.implicit_grad(self._loss_fn)
                gradients_fn = optimizer_lib.get_filtered_grad_fn(gradients_fn)
                grads_and_vars = self.distribution.extended.call_for_each_replica(gradients_fn, args=(ctx, inputs))
                return self._optimizer._distributed_apply(self.distribution, grads_and_vars)
            ctx = self.distribution.extended.experimental_run_steps_on_iterator(step_fn, self._iterator, self._iterations_per_step)
            return ctx.run_op