import multiprocessing
import os
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test
class DTensorTestBackendConfigurator:
    """Configurate test backends."""

    def __init__(self, test_case: tf_test.TestCase):
        self._test_case = test_case

    def tearDown(self):
        if accelerator_util.is_initialized():
            accelerator_util.shutdown_accelerator_system()