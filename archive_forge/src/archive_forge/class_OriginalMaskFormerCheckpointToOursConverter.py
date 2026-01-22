import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple
import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
from PIL import Image
from torch import Tensor, nn
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
from transformers.utils import logging
class OriginalMaskFormerCheckpointToOursConverter:

    def __init__(self, original_model: nn.Module, config: MaskFormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: MaskFormerConfig):
        dst_prefix: str = 'pixel_level_module.encoder'
        src_prefix: str = 'backbone'
        renamed_keys = [(f'{src_prefix}.patch_embed.proj.weight', f'{dst_prefix}.model.embeddings.patch_embeddings.projection.weight'), (f'{src_prefix}.patch_embed.proj.bias', f'{dst_prefix}.model.embeddings.patch_embeddings.projection.bias'), (f'{src_prefix}.patch_embed.norm.weight', f'{dst_prefix}.model.embeddings.norm.weight'), (f'{src_prefix}.patch_embed.norm.bias', f'{dst_prefix}.model.embeddings.norm.bias')]
        num_layers = len(config.backbone_config.depths)
        for layer_idx in range(num_layers):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table')])
                src_att_weight = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
                src_att_bias = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
                dst_state_dict[f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index', f'{dst_prefix}.model.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index')])
            if layer_idx < num_layers - 1:
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.model.encoder.layers.{layer_idx}.downsample.norm.bias')])
            renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.{layer_idx}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.{layer_idx}.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'pixel_level_module.decoder'
        src_prefix: str = 'sem_seg_head.pixel_decoder'
        self.replace_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
            return [(f'{detectron_conv}.weight', f'{mine_conv}.0.weight'), (f'{detectron_conv}.norm.weight', f'{mine_conv}.1.weight'), (f'{detectron_conv}.norm.bias', f'{mine_conv}.1.bias')]
        renamed_keys = [(f'{src_prefix}.mask_features.weight', f'{dst_prefix}.mask_projection.weight'), (f'{src_prefix}.mask_features.bias', f'{dst_prefix}.mask_projection.bias')]
        renamed_keys.extend(rename_keys_for_conv(f'{src_prefix}.layer_4', f'{dst_prefix}.fpn.stem'))
        for src_i, dst_i in zip(range(3, 0, -1), range(0, 3)):
            renamed_keys.extend(rename_keys_for_conv(f'{src_prefix}.adapter_{src_i}', f'{dst_prefix}.fpn.layers.{dst_i}.proj'))
            renamed_keys.extend(rename_keys_for_conv(f'{src_prefix}.layer_{src_i}', f'{dst_prefix}.fpn.layers.{dst_i}.block'))
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def rename_keys_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder'
        src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
        rename_keys = []
        for i in range(self.config.decoder_config.decoder_layers):
            rename_keys.append((f'{src_prefix}.layers.{i}.self_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.self_attn.out_proj.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.self_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.self_attn.out_proj.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.multihead_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.multihead_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.linear1.weight', f'{dst_prefix}.layers.{i}.fc1.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.linear1.bias', f'{dst_prefix}.layers.{i}.fc1.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.linear2.weight', f'{dst_prefix}.layers.{i}.fc2.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.linear2.bias', f'{dst_prefix}.layers.{i}.fc2.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm1.weight', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm1.bias', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm2.weight', f'{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm2.bias', f'{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm3.weight', f'{dst_prefix}.layers.{i}.final_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.layers.{i}.norm3.bias', f'{dst_prefix}.layers.{i}.final_layer_norm.bias'))
        return rename_keys

    def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder'
        src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
        for i in range(self.config.decoder_config.decoder_layers):
            in_proj_weight = src_state_dict.pop(f'{src_prefix}.layers.{i}.self_attn.in_proj_weight')
            in_proj_bias = src_state_dict.pop(f'{src_prefix}.layers.{i}.self_attn.in_proj_bias')
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.q_proj.bias'] = in_proj_bias[:256]
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.k_proj.bias'] = in_proj_bias[256:512]
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.self_attn.v_proj.bias'] = in_proj_bias[-256:]
            in_proj_weight_cross_attn = src_state_dict.pop(f'{src_prefix}.layers.{i}.multihead_attn.in_proj_weight')
            in_proj_bias_cross_attn = src_state_dict.pop(f'{src_prefix}.layers.{i}.multihead_attn.in_proj_bias')
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight'] = in_proj_weight_cross_attn[:256, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias'] = in_proj_bias_cross_attn[:256]
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight'] = in_proj_weight_cross_attn[256:512, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias'] = in_proj_bias_cross_attn[256:512]
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight'] = in_proj_weight_cross_attn[-256:, :]
            dst_state_dict[f'{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias'] = in_proj_bias_cross_attn[-256:]

    def replace_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder'
        src_prefix: str = 'sem_seg_head.predictor.transformer.decoder'
        renamed_keys = self.rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
        renamed_keys.extend([(f'{src_prefix}.norm.weight', f'{dst_prefix}.layernorm.weight'), (f'{src_prefix}.norm.bias', f'{dst_prefix}.layernorm.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module'
        src_prefix: str = 'sem_seg_head.predictor'
        self.replace_detr_decoder(dst_state_dict, src_state_dict)
        renamed_keys = [(f'{src_prefix}.query_embed.weight', f'{dst_prefix}.queries_embedder.weight'), (f'{src_prefix}.input_proj.weight', f'{dst_prefix}.input_projection.weight'), (f'{src_prefix}.input_proj.bias', f'{dst_prefix}.input_projection.bias')]
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_instance_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = ''
        src_prefix: str = 'sem_seg_head.predictor'
        renamed_keys = [(f'{src_prefix}.class_embed.weight', f'{dst_prefix}class_predictor.weight'), (f'{src_prefix}.class_embed.bias', f'{dst_prefix}class_predictor.bias')]
        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend([(f'{src_prefix}.mask_embed.layers.{i}.weight', f'{dst_prefix}mask_embedder.{i}.0.weight'), (f'{src_prefix}.mask_embed.layers.{i}.bias', f'{dst_prefix}mask_embedder.{i}.0.bias')])
        logger.info(f'Replacing keys {pformat(renamed_keys)}')
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, mask_former: MaskFormerModel) -> MaskFormerModel:
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        src_state_dict = self.original_model.state_dict()
        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        logger.info(f'Missed keys are {pformat(dst_state_dict.diff())}')
        logger.info(f'Not copied keys are {pformat(src_state_dict.keys())}')
        logger.info('🙌 Done')
        mask_former.load_state_dict(dst_state_dict)
        return mask_former

    def convert_instance_segmentation(self, mask_former: MaskFormerForInstanceSegmentation) -> MaskFormerForInstanceSegmentation:
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        src_state_dict = self.original_model.state_dict()
        self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)
        mask_former.load_state_dict(dst_state_dict)
        return mask_former

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob('**/*.pkl')
        for checkpoint in checkpoints:
            logger.info(f'💪 Converting {checkpoint.stem}')
            config: Path = config_dir / checkpoint.parents[0].stem / 'swin' / f'{checkpoint.stem}.yaml'
            yield (config, checkpoint)