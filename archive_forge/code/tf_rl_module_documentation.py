import pathlib
from typing import Any, Mapping, Union
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
Saves the weights of this RLModule to the directory dir.

        Args:
            dir: The directory to save the checkpoint to.

        NOTE: For this TfRLModule, we save the weights in the TF checkpoint
            format, so the file name should have no ending and should be a plain string.
            e.g. "my_checkpoint" instead of "my_checkpoint.h5". This method of
            checkpointing saves the module weights as multiple files, so we recommend
            passing a file path relative to a directory, e.g.
            "my_checkpoint/module_state".

        