import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
class CompileLoss(losses_module.Loss):

    def __init__(self, loss, loss_weights=None, reduction='sum_over_batch_size', output_names=None):
        if loss_weights and (not isinstance(loss_weights, (list, tuple, dict))):
            raise ValueError(f'Expected `loss_weights` argument to be a list, tuple, or dict. Received instead: loss_weights={loss_weights} of type {type(loss_weights)}')
        self._user_loss = loss
        self._user_loss_weights = loss_weights
        self.built = False
        self.output_names = output_names
        super().__init__(name='compile_loss', reduction=reduction)

    def build(self, y_true, y_pred):
        if self.output_names:
            output_names = self.output_names
        elif isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all((hasattr(x, '_keras_history') for x in y_pred)):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1
        if output_names:
            num_outputs = len(output_names)
        y_pred = self._flatten_y(y_pred)
        loss = self._user_loss
        loss_weights = self._user_loss_weights
        flat_losses = []
        flat_loss_weights = []
        if isinstance(loss, dict):
            for name in loss.keys():
                if name not in output_names:
                    raise ValueError(f"In the dict argument `loss`, key '{name}' does not correspond to any model output. Received:\nloss={loss}")
        if num_outputs == 1:
            if isinstance(loss, dict):
                loss = tree.flatten(loss)
            if isinstance(loss, list) and len(loss) == 1:
                loss = loss[0]
            if not is_function_like(loss):
                raise ValueError(f'When there is only a single output, the `loss` argument must be a callable. Received instead:\nloss={loss} of type {type(loss)}')
            if isinstance(y_pred, list) and len(y_pred) == 1:
                y_pred = y_pred[0]
        if is_function_like(loss) and tree.is_nested(y_pred):
            loss = tree.map_structure(lambda x: loss, y_pred)
        if is_function_like(loss):
            flat_losses.append(get_loss(loss, y_true, y_pred))
            if loss_weights:
                if not isinstance(loss_weights, float):
                    raise ValueError(f'When there is only a single output, the `loss_weights` argument must be a Python float. Received instead: loss_weights={loss_weights} of type {type(loss_weights)}')
                flat_loss_weights.append(loss_weights)
            else:
                flat_loss_weights.append(1.0)
        elif isinstance(loss, (list, tuple)):
            loss = tree.flatten(loss)
            if len(loss) != len(y_pred):
                raise ValueError(f'For a model with multiple outputs, when providing the `loss` argument as a list, it should have as many entries as the model has outputs. Received:\nloss={loss}\nof length {len(loss)} whereas the model has {len(y_pred)} outputs.')
            if not all((is_function_like(e) for e in loss)):
                raise ValueError(f'For a model with multiple outputs, when providing the `loss` argument as a list, each list entry should be a callable (the loss function corresponding to that output). Received: loss={loss}')
            flat_losses = [get_loss(fn, y_true, y_pred) for fn in loss if fn is not None]
            if loss_weights:
                if not isinstance(loss_weights, (list, tuple)):
                    raise ValueError(f'If the `loss` argument is provided as a list/tuple, the `loss_weight` argument should also be provided as a list/tuple, of equal length. Received: loss_weights={loss_weights}')
                if len(loss_weights) != len(y_pred):
                    raise ValueError(f'For a model with multiple outputs, when providing the `loss_weights` argument as a list, it should have as many entries as the model has outputs. Received: loss_weights={loss_weights} of length {len(loss_weights)} whereas the model has {len(y_pred)} outputs.')
                if not all((isinstance(e, (int, float)) for e in loss_weights)):
                    raise ValueError(f'For a model with multiple outputs, when providing the `loss_weights` argument as a list, each list entry should be a Python int or float (the weighting coefficient corresponding to the loss for that output). Received: loss_weights={loss_weights}')
                flat_loss_weights = list(loss_weights)
            else:
                flat_loss_weights = [1.0 for _ in loss]
        elif isinstance(loss, dict):
            if output_names is None:
                raise ValueError(f'Argument `loss` can only be provided as a dict when the model also returns a dict of outputs. Received loss={loss}')
            for name in loss.keys():
                if isinstance(loss[name], list) and len(loss[name]) == 1:
                    loss[name] = loss[name][0]
                if not is_function_like(loss[name]):
                    raise ValueError(f"For a model with multiple outputs, when providing the `loss` argument as a dict, each dict entry should be a callable (the loss function corresponding to that output). At key '{name}', received invalid type:\n{loss[name]}")
            for name, yt, yp in zip(output_names, y_true, y_pred):
                if name in loss:
                    if loss[name]:
                        flat_losses.append(get_loss(loss[name], yt, yp))
                    else:
                        flat_losses.append(None)
                else:
                    flat_losses.append(None)
            if loss_weights:
                if not isinstance(loss_weights, dict):
                    raise ValueError(f'If the `loss` argument is provided as a dict, the `loss_weight` argument should also be provided as a dict. Received: loss_weights={loss_weights}')
                for name in loss_weights.keys():
                    if name not in output_names:
                        raise ValueError(f"In the dict argument `loss_weights`, key '{name}' does not correspond to any model output. Received: loss_weights={loss_weights}")
                    if not isinstance(loss_weights[name], float):
                        raise ValueError(f"For a model with multiple outputs, when providing the `loss_weights` argument as a dict, each dict entry should be a Python float (the weighting coefficient corresponding to the loss for that output). At key '{name}', received invalid type:\n{loss_weights[name]}")
                for name in output_names:
                    if name in loss_weights:
                        flat_loss_weights.append(loss_weights[name])
                    else:
                        flat_loss_weights.append(1.0)
            else:
                flat_loss_weights = [1.0 for _ in flat_losses]
        self.flat_losses = flat_losses
        self.flat_loss_weights = flat_loss_weights
        self.built = True

    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            return self.call(y_true, y_pred, sample_weight)

    def _flatten_y(self, y):
        if isinstance(y, dict) and self.output_names:
            result = []
            for name in self.output_names:
                if name in y:
                    result.append(y[name])
            return result
        return tree.flatten(y)

    def call(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = self._flatten_y(y_true)
        y_pred = self._flatten_y(y_pred)
        if sample_weight is not None:
            sample_weight = self._flatten_y(sample_weight)
            if len(sample_weight) < len(y_true):
                sample_weight = [sample_weight[0] for _ in range(len(y_true))]
        else:
            sample_weight = [None for _ in y_true]
        loss_values = []
        for loss, y_t, y_p, loss_weight, sample_weight in zip(self.flat_losses, y_true, y_pred, self.flat_loss_weights, sample_weight):
            if loss:
                value = loss_weight * ops.cast(loss(y_t, y_p, sample_weight), dtype=backend.floatx())
                loss_values.append(value)
        if loss_values:
            total_loss = sum(loss_values)
            return total_loss
        return None

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError