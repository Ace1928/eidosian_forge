import json
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
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn
from transformers import (
from transformers.models.mask2former.modeling_mask2former import (
from transformers.utils import logging
class OriginalMask2FormerCheckpointToOursConverter:

    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    def replace_maskformer_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
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

    def replace_swin_backbone(self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig):
        dst_prefix: str = 'pixel_level_module.encoder'
        src_prefix: str = 'backbone'
        renamed_keys = [(f'{src_prefix}.patch_embed.proj.weight', f'{dst_prefix}.embeddings.patch_embeddings.projection.weight'), (f'{src_prefix}.patch_embed.proj.bias', f'{dst_prefix}.embeddings.patch_embeddings.projection.bias'), (f'{src_prefix}.patch_embed.norm.weight', f'{dst_prefix}.embeddings.norm.weight'), (f'{src_prefix}.patch_embed.norm.bias', f'{dst_prefix}.embeddings.norm.bias')]
        for layer_idx in range(len(config.backbone_config.depths)):
            for block_idx in range(config.backbone_config.depths[layer_idx]):
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm1.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_before.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_bias_table', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_bias_table')])
                src_att_weight = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight']
                src_att_bias = src_state_dict[f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias']
                size = src_att_weight.shape[0]
                offset = size // 3
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.weight'] = src_att_weight[:offset, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.query.bias'] = src_att_bias[:offset]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.weight'] = src_att_weight[offset:offset * 2, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.key.bias'] = src_att_bias[offset:offset * 2]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.weight'] = src_att_weight[-offset:, :]
                dst_state_dict[f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.value.bias'] = src_att_bias[-offset:]
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.weight')
                src_state_dict.pop(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.qkv.bias')
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.proj.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.norm2.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.layernorm_after.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc1.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.intermediate.dense.bias'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.weight'), (f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.mlp.fc2.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.output.dense.bias')])
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.blocks.{block_idx}.attn.relative_position_index', f'{dst_prefix}.encoder.layers.{layer_idx}.blocks.{block_idx}.attention.self.relative_position_index')])
            if layer_idx < 3:
                renamed_keys.extend([(f'{src_prefix}.layers.{layer_idx}.downsample.reduction.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.reduction.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.weight', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.weight'), (f'{src_prefix}.layers.{layer_idx}.downsample.norm.bias', f'{dst_prefix}.encoder.layers.{layer_idx}.downsample.norm.bias')])
            renamed_keys.extend([(f'{src_prefix}.norm{layer_idx}.weight', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.weight'), (f'{src_prefix}.norm{layer_idx}.bias', f'{dst_prefix}.hidden_states_norms.stage{layer_idx + 1}.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'pixel_level_module.decoder'
        src_prefix: str = 'sem_seg_head.pixel_decoder'
        self.replace_swin_backbone(dst_state_dict, src_state_dict, self.config)

        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [(f'{src_prefix}.weight', f'{dst_prefix}.weight'), (f'{src_prefix}.bias', f'{dst_prefix}.bias')]

        def rename_keys_for_self_attn(src_prefix: str, dst_prefix: str):
            self_attn_keys = []
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.attention_weights', f'{dst_prefix}.attention_weights'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.output_proj', f'{dst_prefix}.output_proj'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.sampling_offsets', f'{dst_prefix}.sampling_offsets'))
            self_attn_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.value_proj', f'{dst_prefix}.value_proj'))
            return self_attn_keys

        def rename_keys_for_encoder_layer(src_prefix: str, dst_prefix: str):
            encoder_keys = []
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear1', f'{dst_prefix}.fc1'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.linear2', f'{dst_prefix}.fc2'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm1', f'{dst_prefix}.self_attn_layer_norm'))
            encoder_keys.extend(rename_keys_for_weight_bias(f'{src_prefix}.norm2', f'{dst_prefix}.final_layer_norm'))
            encoder_keys.extend(rename_keys_for_self_attn(f'{src_prefix}.self_attn', f'{dst_prefix}.self_attn'))
            return encoder_keys
        renamed_keys = [(f'{src_prefix}.adapter_1.weight', f'{dst_prefix}.adapter_1.0.weight'), (f'{src_prefix}.adapter_1.norm.weight', f'{dst_prefix}.adapter_1.1.weight'), (f'{src_prefix}.adapter_1.norm.bias', f'{dst_prefix}.adapter_1.1.bias')]
        renamed_keys.extend([(f'{src_prefix}.layer_1.weight', f'{dst_prefix}.layer_1.0.weight'), (f'{src_prefix}.layer_1.norm.weight', f'{dst_prefix}.layer_1.1.weight'), (f'{src_prefix}.layer_1.norm.bias', f'{dst_prefix}.layer_1.1.bias')])
        for i in range(3):
            for j in range(2):
                renamed_keys.extend([(f'{src_prefix}.input_proj.{i}.{j}.weight', f'{dst_prefix}.input_projections.{i}.{j}.weight'), (f'{src_prefix}.input_proj.{i}.{j}.bias', f'{dst_prefix}.input_projections.{i}.{j}.bias')])
        renamed_keys.extend([(f'{src_prefix}.transformer.level_embed', f'{dst_prefix}.level_embed')])
        for layer_idx in range(self.config.encoder_layers):
            renamed_keys.extend(rename_keys_for_encoder_layer(f'{src_prefix}.transformer.encoder.layers.{layer_idx}', f'{dst_prefix}.encoder.layers.{layer_idx}'))
        renamed_keys.extend([(f'{src_prefix}.mask_features.weight', f'{dst_prefix}.mask_projection.weight'), (f'{src_prefix}.mask_features.bias', f'{dst_prefix}.mask_projection.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def rename_keys_in_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder'
        src_prefix: str = 'sem_seg_head.predictor'
        rename_keys = []
        for i in range(self.config.decoder_layers - 1):
            rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.self_attn.out_proj.weight'))
            rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.self_attn.out_proj.bias'))
            rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.transformer_self_attention_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.self_attn_layer_norm.bias'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight', f'{dst_prefix}.layers.{i}.cross_attn.in_proj_weight'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias', f'{dst_prefix}.layers.{i}.cross_attn.in_proj_bias'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight', f'{dst_prefix}.layers.{i}.cross_attn.out_proj.weight'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias', f'{dst_prefix}.layers.{i}.cross_attn.out_proj.bias'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.cross_attn_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.transformer_cross_attention_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.cross_attn_layer_norm.bias'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear1.weight', f'{dst_prefix}.layers.{i}.fc1.weight'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear1.bias', f'{dst_prefix}.layers.{i}.fc1.bias'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear2.weight', f'{dst_prefix}.layers.{i}.fc2.weight'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.linear2.bias', f'{dst_prefix}.layers.{i}.fc2.bias'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.norm.weight', f'{dst_prefix}.layers.{i}.final_layer_norm.weight'))
            rename_keys.append((f'{src_prefix}.transformer_ffn_layers.{i}.norm.bias', f'{dst_prefix}.layers.{i}.final_layer_norm.bias'))
        return rename_keys

    def replace_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder'
        src_prefix: str = 'sem_seg_head.predictor'
        renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)
        renamed_keys.extend([(f'{src_prefix}.decoder_norm.weight', f'{dst_prefix}.layernorm.weight'), (f'{src_prefix}.decoder_norm.bias', f'{dst_prefix}.layernorm.bias')])
        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend([(f'{src_prefix}.mask_embed.layers.{i}.weight', f'{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight'), (f'{src_prefix}.mask_embed.layers.{i}.bias', f'{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias')])
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module.decoder.layers'
        src_prefix: str = 'sem_seg_head.predictor'
        for i in range(self.config.decoder_layers - 1):
            in_proj_weight = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight')
            in_proj_bias = src_state_dict.pop(f'{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias')
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.q_proj.weight'] = in_proj_weight[:256, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.q_proj.bias'] = in_proj_bias[:256]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.k_proj.weight'] = in_proj_weight[256:512, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.k_proj.bias'] = in_proj_bias[256:512]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.v_proj.weight'] = in_proj_weight[-256:, :]
            dst_state_dict[f'{dst_prefix}.{i}.self_attn.v_proj.bias'] = in_proj_bias[-256:]

    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = 'transformer_module'
        src_prefix: str = 'sem_seg_head.predictor'
        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)
        renamed_keys = [(f'{src_prefix}.query_embed.weight', f'{dst_prefix}.queries_embedder.weight'), (f'{src_prefix}.query_feat.weight', f'{dst_prefix}.queries_features.weight'), (f'{src_prefix}.level_embed.weight', f'{dst_prefix}.level_embed.weight')]
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = ''
        src_prefix: str = 'sem_seg_head.predictor'
        renamed_keys = [(f'{src_prefix}.class_embed.weight', f'{dst_prefix}class_predictor.weight'), (f'{src_prefix}.class_embed.bias', f'{dst_prefix}class_predictor.bias')]
        logger.info(f'Replacing keys {pformat(renamed_keys)}')
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def convert(self, mask2former: Mask2FormerModel) -> Mask2FormerModel:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()
        self.replace_pixel_module(dst_state_dict, src_state_dict)
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        logger.info(f'Missed keys are {pformat(dst_state_dict.diff())}')
        logger.info(f'Not copied keys are {pformat(src_state_dict.keys())}')
        logger.info('🙌 Done')
        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        mask2former.load_state_dict(state_dict)
        return mask2former

    def convert_universal_segmentation(self, mask2former: Mask2FormerForUniversalSegmentation) -> Mask2FormerForUniversalSegmentation:
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()
        self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)
        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        mask2former.load_state_dict(state_dict)
        return mask2former

    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        checkpoints: List[Path] = checkpoints_dir.glob('**/*.pkl')
        for checkpoint in checkpoints:
            logger.info(f'💪 Converting {checkpoint.stem}')
            dataset_name = checkpoint.parents[2].stem
            if dataset_name == 'ade':
                dataset_name = dataset_name.replace('ade', 'ade20k')
            segmentation_task = checkpoint.parents[1].stem
            config_file_name = f'{checkpoint.parents[0].stem}.yaml'
            config: Path = config_dir / dataset_name / segmentation_task / 'swin' / config_file_name
            yield (config, checkpoint)