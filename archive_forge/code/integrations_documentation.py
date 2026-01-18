from contextlib import contextmanager
import packaging.version
import torch
import transformers
Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing.