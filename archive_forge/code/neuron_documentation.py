import os
import sys
import json
import logging
import subprocess
from typing import Optional, List, Tuple
from ray._private.accelerators.accelerator import AcceleratorManager
Set the NEURON_RT_VISIBLE_CORES environment variable based on
        given visible_neuron_core_ids.

        Args:
            visible_neuron_core_ids (List[str]): List of int representing core IDs.
        