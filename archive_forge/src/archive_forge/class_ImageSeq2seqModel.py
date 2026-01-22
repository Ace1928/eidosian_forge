from enum import Enum
from functools import reduce
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from parlai.agents.transformer.modules import (
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
class ImageSeq2seqModel(TransformerGeneratorModel):
    """
    ImageSeq2seqModel.

    Just TGA that can encode image with encoder.
    """

    def __init__(self, opt: Opt, dictionary: DictionaryAgent):
        if opt.get('n_positions'):
            n_positions = opt['n_positions']
        else:
            n_positions = max(opt.get('truncate') or 0, opt.get('text_truncate') or 0, opt.get('label_truncate') or 0)
            if n_positions == 0:
                n_positions = 1024
        super().__init__(opt, dictionary)
        self.encoder = ContextWithImageEncoder(n_heads=opt['n_heads'], n_layers=opt['n_layers'], embedding_size=opt['embedding_size'], ffn_size=opt['ffn_size'], vocabulary_size=len(dictionary), embedding=self.embeddings, dropout=opt['dropout'], attention_dropout=opt['attention_dropout'], relu_dropout=opt['relu_dropout'], padding_idx=self.pad_idx, learn_positional_embeddings=opt['learn_positional_embeddings'], embeddings_scale=opt['embeddings_scale'], n_positions=n_positions, n_segments=opt.get('n_segments', 0), activation=opt['activation'], variant=opt['variant'], output_scaling=opt['output_scaling'], image_encoder_num_layers=opt['image_encoder_num_layers'], image_features_dim=opt['image_features_dim'], fusion=opt['image_fusion_type'], n_image_tokens=opt.get('n_image_tokens', 1), n_image_channels=opt.get('n_image_channels', 1))