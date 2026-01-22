import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src import optimizers
from keras.src.dtensor import utils as dtensor_utils
from keras.src.optimizers import optimizer
from keras.src.optimizers import utils as optimizer_utils
from keras.src.optimizers.legacy import optimizer_v2
from keras.src.saving import serialization_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.util.tf_export import keras_export
class LossScaleOptimizerV3(tf.__internal__.tracking.DelegatingTrackableMixin, optimizer.Optimizer, BaseLossScaleOptimizer):
    """An optimizer that applies loss scaling to prevent numeric underflow.

    This is a copy of the `mixed_precision.LossScaleOptimizer` class
    defined above, except it subclasses and wraps the new experimental Optimizer
    class instead of the `tf.keras.optimizers.Optimizer` class. Some of the
    methods this class defines and calls are different compared to
    LossScaleOptimizer due to the differences between the two Optimizer base
    classes. Additionally, this class does not support the legacy graph mode,
    but LossScaleOptimizer does.

    Since the new experimental Optimizer does not have a hyperparameter concept,
    LossScaleOptimizerV3 does not delegate arbitrary hyperparameter accesses to
    the inner optimizer, unlike LossScaleOptimizer. LossScaleOptimizerV3 does
    delegate the "learning_rate" attribute, however.
    """

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(self, inner_optimizer, dynamic=True, initial_scale=None, dynamic_growth_steps=None):
        if not isinstance(inner_optimizer, optimizer.Optimizer):
            if isinstance(inner_optimizer, optimizer_v2.OptimizerV2):
                raise TypeError(f'You passed a `tf.keras.optimizers.Optimizer` instance to LossScaleOptimizerV3, but only the new experimental optimizer defined in keras/optimizer_expeirmental/optimizer.py can be passed. Please use `tf.keras.mixed_precision.LossScaleOptimizer` instead of LossScaleOptimizerV3, as the former supports `tf.keras.optimizers.Optimizer`s. Got optimizer: {inner_optimizer}')
            raise TypeError(f'"inner_optimizer" must be an instance of Optimizer, but got: {inner_optimizer}.')
        if not isinstance(dynamic, bool):
            raise TypeError(f'"dynamic" argument to LossScaleOptimizer.__init__ must be a bool, but got: {repr(dynamic)}')
        if isinstance(inner_optimizer, LossScaleOptimizerV3):
            raise TypeError(f'LossScaleOptimizer cannot wrap another LossScaleOptimizer, but got: {inner_optimizer}')
        _raise_if_strategy_unsupported()
        if getattr(inner_optimizer, '_is_wrapped_by_loss_scale_optimizer', False):
            raise ValueError('"inner_optimizer" is already wrapped by a LossScaleOptimizer. An optimizer can only be wrapped by a single LossScaleOptimizer')
        self._optimizer = inner_optimizer
        self._optimizer._is_wrapped_by_loss_scale_optimizer = True
        tf.__internal__.tracking.DelegatingTrackableMixin.__init__(self, self._optimizer)
        if dynamic:
            if initial_scale is None:
                initial_scale = _DEFAULT_INITIAL_SCALE
            if dynamic_growth_steps is None:
                dynamic_growth_steps = _DEFAULT_GROWTH_STEPS
            self._loss_scale = _DynamicLossScaleState(initial_scale, dynamic_growth_steps, multiplier=2)
            self._track_trackable(self._loss_scale, 'loss_scale')
        else:
            if initial_scale is None:
                raise ValueError('"initial_scale" must be specified if "dynamic" is False')
            self._loss_scale = float(initial_scale)
            if dynamic_growth_steps is not None:
                raise ValueError(f'"dynamic_growth_steps" must be None if "dynamic" is False, but got: {dynamic_growth_steps}')
        self._loss_has_been_scaled = False
        self._gradients_have_been_unscaled = False

    @property
    def dynamic(self):
        return isinstance(self._loss_scale, _DynamicLossScaleState)

    @property
    def loss_scale(self):
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return tf.convert_to_tensor(self._loss_scale.current_loss_scale)
        else:
            return tf.convert_to_tensor(self._loss_scale)

    @property
    def dynamic_counter(self):
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.counter
        else:
            return None

    @property
    def initial_scale(self):
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.initial_loss_scale
        else:
            return self._loss_scale

    @property
    def dynamic_growth_steps(self):
        if isinstance(self._loss_scale, _DynamicLossScaleState):
            return self._loss_scale.growth_steps
        else:
            return None

    @property
    def inner_optimizer(self):
        return self._optimizer

    def get_scaled_loss(self, loss):
        self._loss_has_been_scaled = True
        if callable(loss):

            def new_loss():
                loss_val = loss()
                return loss_val * tf.cast(self.loss_scale, loss_val.dtype)
            return new_loss
        else:
            return loss * tf.cast(self.loss_scale, loss.dtype)

    def get_unscaled_gradients(self, grads):
        self._gradients_have_been_unscaled = True
        loss_scale_reciprocal = 1.0 / self.loss_scale
        return [_multiply_gradient(g, loss_scale_reciprocal) if g is not None else None for g in grads]

    def compute_gradients(self, loss, var_list, tape=None):
        tape = tf.GradientTape() if tape is None else tape
        with tape:
            loss = self.get_scaled_loss(loss)
        grads_and_vars = self._optimizer.compute_gradients(loss, var_list, tape=tape)
        grads = [g for g, _ in grads_and_vars]
        weights = [v for _, v in grads_and_vars]
        unscaled_grads = self.get_unscaled_gradients(grads)
        return list(zip(unscaled_grads, weights))

    def apply_gradients(self, grads_and_vars, skip_gradients_aggregation=False, **kwargs):
        grads_and_vars = list(grads_and_vars)
        grads, trainable_variables = zip(*grads_and_vars)
        with tf.init_scope():
            self.build(trainable_variables)
        if tf.distribute.in_cross_replica_context():
            raise ValueError('apply_gradients() must be called in a replica context.')
        _raise_if_strategy_unsupported()
        _maybe_warn_about_scaling(self._loss_has_been_scaled, self._gradients_have_been_unscaled)
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        experimental_aggregate_gradients = kwargs.pop('experimental_aggregate_gradients', True)
        run_with_dtensor = self._optimizer._run_with_dtensor or self._optimizer._mesh
        if not skip_gradients_aggregation and experimental_aggregate_gradients and (not run_with_dtensor):
            grads_and_vars = self._optimizer.aggregate_gradients(grads_and_vars)
        grads_and_vars = tuple(grads_and_vars)
        grads = [g for g, _ in grads_and_vars]
        wrapped_vars = _UnwrapPreventer([v for _, v in grads_and_vars])

        def do_not_apply_fn():
            self._optimizer.iterations.assign_add(1, read_value=False)

        def _if_should_apply_grads(grads):
            if isinstance(self._loss_scale, _DynamicLossScaleState):
                _, should_apply_grad = self._loss_scale.update(grads)
                return should_apply_grad
            else:
                return True
        if tf.__internal__.distribute.strategy_supports_no_merge_call():
            should_apply_grads = _if_should_apply_grads(grads)

            def apply_fn():
                return self._apply_gradients(grads, wrapped_vars)
            tf.__internal__.smart_cond.smart_cond(should_apply_grads, apply_fn, do_not_apply_fn)
        else:

            def _apply_gradients_cross_replica(distribution, grads, wrapped_vars):
                should_apply_grads = _if_should_apply_grads(grads)

                def apply_fn():
                    distribution.extended.call_for_each_replica(self._apply_gradients, args=(grads, wrapped_vars))
                tf.__internal__.smart_cond.smart_cond(should_apply_grads, apply_fn, do_not_apply_fn)
            tf.distribute.get_replica_context().merge_call(_apply_gradients_cross_replica, args=(grads, wrapped_vars))

    def _apply_gradients(self, grads, wrapped_vars):
        self._optimizer.apply_gradients(list(zip(grads, wrapped_vars.value)), skip_gradients_aggregation=True)

    def get_config(self):
        serialized_optimizer = optimizers.serialize(self._optimizer)
        return {'inner_optimizer': serialized_optimizer, 'dynamic': self.dynamic, 'initial_scale': self.initial_scale, 'dynamic_growth_steps': self.dynamic_growth_steps}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        if isinstance(config['inner_optimizer'], optimizer.Optimizer):
            inner_optimizer = config['inner_optimizer']
        else:
            inner_optimizer = optimizers.deserialize(config['inner_optimizer'], custom_objects=custom_objects, use_legacy_optimizer=False)
        del config['inner_optimizer']
        return cls(inner_optimizer, **config)

    @property
    def iterations(self):
        return self._optimizer.iterations

    @iterations.setter
    def iterations(self, variable):
        self._optimizer.iterations = variable

    @property
    def variables(self):
        return self._optimizer.variables

    def build(self, var_list):
        return self._optimizer.build(var_list)

    @property
    def learning_rate(self):
        return self._optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer.learning_rate = learning_rate

    @property
    def use_ema(self):
        return self._optimizer.use_ema

    @use_ema.setter
    def use_ema(self, use_ema):
        self._optimizer.use_ema = use_ema

    @property
    def ema_momentum(self):
        return self._optimizer.ema_momentum

    @ema_momentum.setter
    def ema_momentum(self, ema_momentum):
        self._optimizer.ema_momentum = ema_momentum

    def finalize_variable_values(self, var_list):
        self._optimizer.finalize_variable_values(var_list)