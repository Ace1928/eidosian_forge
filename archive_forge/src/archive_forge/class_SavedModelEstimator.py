from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import tensorflow as tf
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.experimental.SavedModelEstimator')
class SavedModelEstimator(estimator_lib.EstimatorV2):
    """Create an Estimator from a SavedModel.

  Only SavedModels exported with
  `tf.estimator.Estimator.experimental_export_all_saved_models()` or
  `tf.estimator.Estimator.export_saved_model()` are supported for this class.

  Example with `tf.estimator.DNNClassifier`:

  **Step 1: Create and train DNNClassifier.**

  ```python
  feature1 = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          key='feature1', vocabulary_list=('green', 'yellow')), dimension=1)
  feature2 = tf.feature_column.numeric_column(key='feature2', default_value=0.0)

  classifier = tf.estimator.DNNClassifier(
      hidden_units=[4,2], feature_columns=[feature1, feature2])

  def input_fn():
    features = {'feature1': tf.constant(['green', 'green', 'yellow']),
                'feature2': tf.constant([3.5, 4.2, 6.1])}
    label = tf.constant([1., 0., 0.])
    return tf.data.Dataset.from_tensors((features, label)).repeat()

  classifier.train(input_fn=input_fn, steps=10)
  ```

  **Step 2: Export classifier.**
  First, build functions that specify the expected inputs.

  ```python
  # During train and evaluation, both the features and labels should be defined.
  supervised_input_receiver_fn = (
      tf.estimator.experimental.build_raw_supervised_input_receiver_fn(
          {'feature1': tf.placeholder(dtype=tf.string, shape=[None]),
           'feature2': tf.placeholder(dtype=tf.float32, shape=[None])},
          tf.placeholder(dtype=tf.float32, shape=[None])))

  # During predict mode, expect to receive a `tf.Example` proto, so a parsing
  # function is used.
  serving_input_receiver_fn = (
      tf.estimator.export.build_parsing_serving_input_receiver_fn(
          tf.feature_column.make_parse_example_spec([feature1, feature2])))
  ```

  Next, export the model as a SavedModel. A timestamped directory will be
  created (for example `/tmp/export_all/1234567890`).

  ```python
  # Option 1: Save all modes (train, eval, predict)
  export_dir = classifier.experimental_export_all_saved_models(
      '/tmp/export_all',
      {tf.estimator.ModeKeys.TRAIN: supervised_input_receiver_fn,
       tf.estimator.ModeKeys.EVAL: supervised_input_receiver_fn,
       tf.estimator.ModeKeys.PREDICT: serving_input_receiver_fn})

  # Option 2: Only export predict mode
  export_dir = classifier.export_saved_model(
      '/tmp/export_predict', serving_input_receiver_fn)
  ```

  **Step 3: Create a SavedModelEstimator from the exported SavedModel.**

  ```python
  est = tf.estimator.experimental.SavedModelEstimator(export_dir)

  # If all modes were exported, you can immediately evaluate and predict, or
  # continue training. Otherwise only predict is available.
  eval_results = est.evaluate(input_fn=input_fn, steps=1)
  print(eval_results)

  est.train(input_fn=input_fn, steps=20)

  def predict_input_fn():
    example = tf.train.Example()
    example.features.feature['feature1'].bytes_list.value.extend(['yellow'])
    example.features.feature['feature2'].float_list.value.extend([1.])
    return {'inputs':tf.constant([example.SerializeToString()])}

  predictions = est.predict(predict_input_fn)
  print(next(predictions))
  ```
  """

    def __init__(self, saved_model_dir, model_dir=None):
        """Initialize a SavedModelEstimator.

    The SavedModelEstimator loads its model function and variable values from
    the graphs defined in the SavedModel. There is no option to pass in
    `RunConfig` or `params` arguments, because the model function graph is
    defined statically in the SavedModel.

    Args:
      saved_model_dir: Directory containing SavedModel protobuf and subfolders.
      model_dir: Directory to save new checkpoints during training.

    Raises:
      NotImplementedError: If a DistributionStrategy is defined in the config.
        Unless the SavedModelEstimator is subclassed, this shouldn't happen.
    """
        super(SavedModelEstimator, self).__init__(model_fn=self._model_fn_from_saved_model, model_dir=model_dir)
        if self._train_distribution or self._eval_distribution:
            raise NotImplementedError('SavedModelEstimator currently does not support DistributionStrategy.')
        self.saved_model_dir = saved_model_dir
        self.saved_model_loader = loader_impl.SavedModelLoader(saved_model_dir)
        self._available_modes = self._extract_available_modes()

    def _extract_available_modes(self):
        """Return list of modes found in SavedModel."""
        available_modes = []
        tf.compat.v1.logging.info('Checking available modes for SavedModelEstimator.')
        for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
            try:
                self._get_meta_graph_def_for_mode(mode)
            except RuntimeError:
                tf.compat.v1.logging.warn('%s mode not found in SavedModel.' % mode)
                continue
            if self._get_signature_def_for_mode(mode) is not None:
                available_modes.append(mode)
        tf.compat.v1.logging.info('Available modes for Estimator: %s' % available_modes)
        return available_modes

    def _validate_mode(self, mode):
        """Make sure that mode can be run using the SavedModel."""
        if mode not in self._available_modes:
            raise RuntimeError('%s mode is not available in the SavedModel. Use saved_model_cli to check that the Metagraph for this mode has been exported.' % mode)

    def _get_meta_graph_def_for_mode(self, mode):
        tags = export_lib.EXPORT_TAG_MAP[mode]
        return self.saved_model_loader.get_meta_graph_def_from_tags(tags)

    def _get_signature_def_for_mode(self, mode):
        meta_graph_def = self._get_meta_graph_def_for_mode(mode)
        if mode == ModeKeys.PREDICT:
            sig_def_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        else:
            sig_def_key = mode
        if sig_def_key not in meta_graph_def.signature_def:
            tf.compat.v1.logging.warn('Metagraph for mode %s was found, but SignatureDef with key "%s" is missing.' % (mode, sig_def_key))
            return None
        return meta_graph_def.signature_def[sig_def_key]

    def _get_saver_def_from_mode(self, mode):
        meta_graph_def = self._get_meta_graph_def_for_mode(mode)
        return meta_graph_def.saver_def

    def _create_and_assert_global_step(self, graph):
        return None

    def _model_fn_from_saved_model(self, features, labels, mode):
        """Load a SavedModel graph and return an EstimatorSpec."""
        self._validate_mode(mode)
        g = tf.compat.v1.get_default_graph()
        if tf.compat.v1.train.get_global_step(g) is not None:
            raise RuntimeError('Graph must not contain a global step tensor before the SavedModel is loaded. Please make sure that the input function does not create a global step.')
        signature_def = self._get_signature_def_for_mode(mode)
        input_map = _generate_input_map(signature_def, features, labels)
        output_tensor_names = [value.name for value in six.itervalues(signature_def.outputs)]
        tags = export_lib.EXPORT_TAG_MAP[mode]
        _, output_tensors = self.saved_model_loader.load_graph(g, tags, input_map=input_map, return_elements=output_tensor_names)
        saver_obj = tf.compat.v1.train.Saver(saver_def=self._get_saver_def_from_mode(mode))
        init_fn = None
        if not super(SavedModelEstimator, self).latest_checkpoint():
            init_fn = self._restore_from_saver
        meta_graph_def = self._get_meta_graph_def_for_mode(mode)
        asset_tensors_dictionary = loader_impl.get_asset_tensors(self.saved_model_loader.export_dir, meta_graph_def, import_scope=None)
        scaffold = tf.compat.v1.train.Scaffold(local_init_op=loader_impl._get_main_op_tensor(meta_graph_def), local_init_feed_dict=asset_tensors_dictionary, saver=saver_obj, init_fn=init_fn)
        global_step_tensor = tf.compat.v1.train.get_global_step(g)
        tf.compat.v1.train.assert_global_step(global_step_tensor)
        output_map = dict(zip(output_tensor_names, output_tensors))
        outputs = {key: output_map[value.name] for key, value in six.iteritems(signature_def.outputs)}
        loss, predictions, metrics = _validate_and_extract_outputs(mode, outputs, signature_def.method_name)
        train_op = tf.compat.v1.get_collection(constants.TRAIN_OP_KEY)
        if len(train_op) > 1:
            raise RuntimeError('Multiple ops found in the train_op collection.')
        train_op = None if not train_op else train_op[0]
        _clear_saved_model_collections()
        return model_fn_lib.EstimatorSpec(scaffold=scaffold, mode=mode, loss=loss, train_op=train_op, predictions=predictions, eval_metric_ops=metrics)

    def _restore_from_saver(self, scaffold, session):
        return scaffold.saver.restore(session, _get_saved_model_ckpt(self.saved_model_dir))

    def latest_checkpoint(self):
        """Returns the filename of the latest saved checkpoint.

    Returns:
      Filename of latest checkpoint in `model_dir`. If no checkpoints are found
      in `model_dir`, then the path to the SavedModel checkpoint is returned.
    """
        return super(SavedModelEstimator, self).latest_checkpoint() or _get_saved_model_ckpt(self.saved_model_dir)