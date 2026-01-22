import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def _load_text_encoder_state(self):
    try:
        state_file = self.opt.get('load_encoder_from')
        model = torch.load(state_file)
        states = model['model']
        self.text_encoder.load_state_dict(states)
    except Exception as e:
        print('WARNING: Cannot load transformer state; please make sure specified file is a dictionary with the states in `model`. Additionally, make sure that the appropriate options are specified. Error: {}'.format(e))