from tensorflow.python.util.tf_export import tf_export
from tensorflow.python import pywrap_tfe
def to_string(self):
    if self._as_string is None:
        self._as_string = self._components_to_string(job=self.job, replica=self.replica, task=self.task, device_type=self.device_type, device_index=self.device_index)
    return self._as_string