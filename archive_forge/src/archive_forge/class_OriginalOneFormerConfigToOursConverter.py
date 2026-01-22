import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging
class OriginalOneFormerConfigToOursConverter:

    def __call__(self, original_config: object, is_swin: bool) -> OneFormerConfig:
        model = original_config.MODEL
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])
        id2label = dict(enumerate(dataset_catalog.stuff_classes))
        label2id = {label: idx for idx, label in id2label.items()}
        if is_swin:
            if model.SWIN.EMBED_DIM == 96:
                backbone_config = SwinConfig.from_pretrained('microsoft/swin-tiny-patch4-window7-224', drop_path_rate=model.SWIN.DROP_PATH_RATE, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
            elif model.SWIN.EMBED_DIM == 192:
                backbone_config = SwinConfig.from_pretrained('microsoft/swin-large-patch4-window12-384', drop_path_rate=model.SWIN.DROP_PATH_RATE, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
            else:
                raise ValueError(f'embed dim {model.SWIN.EMBED_DIM} not supported for Swin!')
        else:
            backbone_config = DinatConfig.from_pretrained('shi-labs/dinat-large-11x11-in22k-in1k-384', dilations=model.DiNAT.DILATIONS, kernel_size=model.DiNAT.KERNEL_SIZE, out_features=['stage1', 'stage2', 'stage3', 'stage4'])
        config: OneFormerConfig = OneFormerConfig(backbone_config=backbone_config, output_attentions=True, output_hidden_states=True, return_dict=True, ignore_value=model.SEM_SEG_HEAD.IGNORE_VALUE, num_classes=model.SEM_SEG_HEAD.NUM_CLASSES, num_queries=model.ONE_FORMER.NUM_OBJECT_QUERIES, no_object_weight=model.ONE_FORMER.NO_OBJECT_WEIGHT, class_weight=model.ONE_FORMER.CLASS_WEIGHT, mask_weight=model.ONE_FORMER.MASK_WEIGHT, dice_weight=model.ONE_FORMER.DICE_WEIGHT, contrastive_weight=model.ONE_FORMER.CONTRASTIVE_WEIGHT, contrastive_temperature=model.ONE_FORMER.CONTRASTIVE_TEMPERATURE, train_num_points=model.ONE_FORMER.TRAIN_NUM_POINTS, oversample_ratio=model.ONE_FORMER.OVERSAMPLE_RATIO, importance_sample_ratio=model.ONE_FORMER.IMPORTANCE_SAMPLE_RATIO, init_std=0.02, init_xavier_std=1.0, layer_norm_eps=1e-05, is_training=False, use_auxiliary_loss=model.ONE_FORMER.DEEP_SUPERVISION, output_auxiliary_logits=True, strides=[4, 8, 16, 32], task_seq_len=original_config.INPUT.TASK_SEQ_LEN, max_seq_len=original_config.INPUT.MAX_SEQ_LEN, text_encoder_width=model.TEXT_ENCODER.WIDTH, text_encoder_context_length=model.TEXT_ENCODER.CONTEXT_LENGTH, text_encoder_num_layers=model.TEXT_ENCODER.NUM_LAYERS, text_encoder_vocab_size=model.TEXT_ENCODER.VOCAB_SIZE, text_encoder_proj_layers=model.TEXT_ENCODER.PROJ_NUM_LAYERS, text_encoder_n_ctx=model.TEXT_ENCODER.N_CTX, conv_dim=model.SEM_SEG_HEAD.CONVS_DIM, mask_dim=model.SEM_SEG_HEAD.MASK_DIM, hidden_dim=model.ONE_FORMER.HIDDEN_DIM, norm=model.SEM_SEG_HEAD.NORM, encoder_layers=model.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS, encoder_feedforward_dim=1024, decoder_layers=model.ONE_FORMER.DEC_LAYERS, use_task_norm=model.ONE_FORMER.USE_TASK_NORM, num_attention_heads=model.ONE_FORMER.NHEADS, dropout=model.ONE_FORMER.DROPOUT, dim_feedforward=model.ONE_FORMER.DIM_FEEDFORWARD, pre_norm=model.ONE_FORMER.PRE_NORM, enforce_input_proj=model.ONE_FORMER.ENFORCE_INPUT_PROJ, query_dec_layers=model.ONE_FORMER.CLASS_DEC_LAYERS, common_stride=model.SEM_SEG_HEAD.COMMON_STRIDE, id2label=id2label, label2id=label2id)
        return config