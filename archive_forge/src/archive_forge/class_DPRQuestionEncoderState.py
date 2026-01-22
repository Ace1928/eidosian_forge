import argparse
import collections
from pathlib import Path
import torch
from torch.serialization import default_restore_location
from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRReader
class DPRQuestionEncoderState(DPRState):

    def load_dpr_model(self):
        model = DPRQuestionEncoder(DPRConfig(**BertConfig.get_config_dict('google-bert/bert-base-uncased')[0]))
        print(f'Loading DPR biencoder from {self.src_file}')
        saved_state = load_states_from_checkpoint(self.src_file)
        encoder, prefix = (model.question_encoder, 'question_model.')
        state_dict = {'bert_model.embeddings.position_ids': model.question_encoder.bert_model.embeddings.position_ids}
        for key, value in saved_state.model_dict.items():
            if key.startswith(prefix):
                key = key[len(prefix):]
                if not key.startswith('encode_proj.'):
                    key = 'bert_model.' + key
                state_dict[key] = value
        encoder.load_state_dict(state_dict)
        return model