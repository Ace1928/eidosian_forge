from parlai.agents.transformer.transformer import TransformerRankerAgent
from .wizard_dict import WizardDictAgent
import numpy as np
import torch

        Return opt and model states.

        Override this method from TorchAgent to allow us to load partial weights from
        pre-trained models.
        