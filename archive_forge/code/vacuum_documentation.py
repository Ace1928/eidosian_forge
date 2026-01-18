import os
import torch
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.torch import atomic_save
import parlai.utils.pickle
import parlai.utils.logging as logging

Reduces the size of a model file by stripping the optimizer.

Assumes we are working with a TorchAgent
