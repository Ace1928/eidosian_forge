from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _SavedModelExporter(Exporter):
    """This class exports the serving graph and checkpoints.

     This class provides a basic exporting functionality and serves as a
     foundation for specialized `Exporter`s.
  """

    def __init__(self, name, serving_input_receiver_fn, assets_extra=None, as_text=False):
        """Create an `Exporter` to use with `tf.estimator.EvalSpec`.

    Args:
      name: unique name of this `Exporter` that is going to be used in the
        export path.
      serving_input_receiver_fn: a function that takes no arguments and returns
        a `ServingInputReceiver`.
      assets_extra: An optional dict specifying how to populate the assets.extra
        directory within the exported SavedModel.  Each key should give the
        destination path (including the filename) relative to the assets.extra
        directory.  The corresponding value gives the full path of the source
        file to be copied.  For example, the simple case of copying a single
        file without renaming it is specified as
        `{'my_asset_file.txt': '/path/to/my_asset_file.txt'}`.
      as_text: whether to write the SavedModel proto in text format. Defaults to
        `False`.

    Raises:
      ValueError: if any arguments is invalid.
    """
        self._name = name
        self._serving_input_receiver_fn = serving_input_receiver_fn
        self._assets_extra = assets_extra
        self._as_text = as_text

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        del is_the_final_export
        export_result = estimator.export_saved_model(export_path, self._serving_input_receiver_fn, assets_extra=self._assets_extra, as_text=self._as_text, checkpoint_path=checkpoint_path)
        return export_result