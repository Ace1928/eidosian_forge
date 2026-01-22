from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.add_metrics')
def add_metrics(estimator, metric_fn):
    """Creates a new `tf.estimator.Estimator` which has given metrics.

  Example:

  ```python
    def my_auc(labels, predictions):
      auc_metric = tf.keras.metrics.AUC(name="my_auc")
      auc_metric.update_state(y_true=labels, y_pred=predictions['logistic'])
      return {'auc': auc_metric}

    estimator = tf.estimator.DNNClassifier(...)
    estimator = tf.estimator.add_metrics(estimator, my_auc)
    estimator.train(...)
    estimator.evaluate(...)
  ```
  Example usage of custom metric which uses features:

  ```python
    def my_auc(labels, predictions, features):
      auc_metric = tf.keras.metrics.AUC(name="my_auc")
      auc_metric.update_state(y_true=labels, y_pred=predictions['logistic'],
                              sample_weight=features['weight'])
      return {'auc': auc_metric}

    estimator = tf.estimator.DNNClassifier(...)
    estimator = tf.estimator.add_metrics(estimator, my_auc)
    estimator.train(...)
    estimator.evaluate(...)
  ```

  Args:
    estimator: A `tf.estimator.Estimator` object.
    metric_fn: A function which should obey the following signature:
      - Args: can only have following four arguments in any order:
        * predictions: Predictions `Tensor` or dict of `Tensor` created by given
          `estimator`.
        * features: Input `dict` of `Tensor` objects created by `input_fn` which
          is given to `estimator.evaluate` as an argument.
        * labels:  Labels `Tensor` or dict of `Tensor` created by `input_fn`
          which is given to `estimator.evaluate` as an argument.
        * config: config attribute of the `estimator`.
       - Returns: Dict of metric results keyed by name. Final metrics are a
         union of this and `estimator's` existing metrics. If there is a name
         conflict between this and `estimator`s existing metrics, this will
         override the existing one. The values of the dict are the results of
         calling a metric function, namely a `(metric_tensor, update_op)` tuple.

  Returns:
      A new `tf.estimator.Estimator` which has a union of original metrics with
        given ones.
  """
    _verify_metric_fn_args(metric_fn)

    def new_model_fn(features, labels, mode, config):
        spec = estimator.model_fn(features, labels, mode, config)
        if mode != ModeKeys.EVAL:
            return spec
        new_metrics = _call_metric_fn(metric_fn, features, labels, spec.predictions, config)
        all_metrics = spec.eval_metric_ops or {}
        all_metrics.update(new_metrics)
        return spec._replace(eval_metric_ops=all_metrics)
    return estimator_lib.Estimator(model_fn=new_model_fn, model_dir=estimator.model_dir, config=estimator.config, warm_start_from=estimator._warm_start_settings)