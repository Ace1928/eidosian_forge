import os
import unittest
import contextlib
import tempfile
import shutil
import io
import signal
from typing import Tuple, Dict, Any
from parlai.core.opt import Opt
import parlai.utils.logging as logging
class AutoTeacherTest:

    def _run_display_data(self, datatype, **kwargs):
        import parlai.scripts.display_data as dd
        dd.DisplayData.main(task=self.task, datatype=datatype, display_verbose=True, **kwargs)

    def test_train(self):
        """
        Test --datatype train.
        """
        return self._run_display_data('train')

    def test_train_stream(self):
        """
        Test --datatype train:stream.
        """
        return self._run_display_data('train:stream')

    def test_train_stream_ordered(self):
        """
        Test --datatype train:stream:ordered.
        """
        return self._run_display_data('train:stream:ordered')

    def test_valid(self):
        """
        Test --datatype valid.
        """
        return self._run_display_data('valid')

    def test_valid_stream(self):
        """
        Test --datatype valid:stream.
        """
        return self._run_display_data('valid:stream')

    def test_test(self):
        """
        Test --datatype test.
        """
        return self._run_display_data('test')

    def test_test_stream(self):
        """
        Test --datatype test:stream.
        """
        return self._run_display_data('test:stream')

    def test_bs2_train(self):
        """
        Test --datatype train.
        """
        return self._run_display_data('train', batchsize=2)

    def test_bs2_train_stream(self):
        """
        Test --datatype train:stream.
        """
        return self._run_display_data('train:stream', batchsize=2)

    def test_bs2_train_stream_ordered(self):
        """
        Test --datatype train:stream:ordered.
        """
        return self._run_display_data('train:stream:ordered', batchsize=2)

    def test_bs2_valid(self):
        """
        Test --datatype valid.
        """
        return self._run_display_data('valid', batchsize=2)

    def test_bs2_valid_stream(self):
        """
        Test --datatype valid:stream.
        """
        return self._run_display_data('valid:stream', batchsize=2)

    def test_bs2_test(self):
        """
        Test --datatype test.
        """
        return self._run_display_data('test', batchsize=2)

    def test_bs2_test_stream(self):
        """
        Test --datatype test:stream.
        """
        return self._run_display_data('test:stream', batchsize=2)