import torch
import random
import pytest
from unittest.mock import MagicMock
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed
from .utils import (create_execute_model_data, create_worker,
Verify the multi-step worker produces the same output as the normal
    worker when num_steps > 1. This test runs the multi-step worker once, and
    then runs the worker num_steps times, and compares the output.
    