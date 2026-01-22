from tensorflow.python.util.tf_export import tf_export
def _no_dependency(self, *args, **kwargs):
    return self._trackable._no_dependency(*args, **kwargs)