import torch
import trajectory.utils as utils
from transformers import TrajectoryTransformerModel

    To run this script you will need to install the original repository to run the original model. You can find it
    here: https://github.com/jannerm/trajectory-transformer From this repository code you can also download the
    original pytorch checkpoints.

    Run with the command:

    ```sh
    >>> python convert_trajectory_transformer_original_pytorch_checkpoint_to_pytorch.py --dataset <dataset_name>
    ...     --gpt_loadpath <path_to_original_pytorch_checkpoint>
    ```
    