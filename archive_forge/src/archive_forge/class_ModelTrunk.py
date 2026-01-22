from enum import Enum
from typing import Dict, Union
import pytorch_lightning as pl
import torch
import torch.nn as nn
from xformers.components import build_attention
from xformers.components.multi_head_dispatch import MultiHeadDispatchConfig
from xformers.factory import xFormer, xFormerConfig, xFormerEncoderConfig
from xformers.utils import generate_matching_config
class ModelTrunk(pl.LightningModule):

    def __init__(self, config, model_name):
        super().__init__()
        config_model = config['model']
        self.config_training = config['training']
        self.enable_amp = config['training']['mixed_precision']
        self.pooling_mode = Pooling(config_model['pooling_mode'])
        self.vocab_size = config_model['common']['vocab_size']
        self.config_model = patch_model_config(config_model, model_name)
        self.model = xFormer.from_config(xFormerConfig(config_model['xformer']))
        self.norm = nn.LayerNorm(self.config_model['common']['dim_model'])
        ff_config = self.config_model['xformer'][0]['feedforward_config']
        self.dim_mlp = self.config_model['common']['dim_model'] * ff_config['hidden_layer_multiplier']

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> PLOutput:
        outputs = self(**batch)
        self.logger.log_metrics({f'train_{k}': v for k, v in outputs.items()})
        self.log('train_accu', outputs['accu'], sync_dist=True)
        return outputs

    def training_epoch_end(self, outputs):
        logs = self.eval_epoch_end(outputs)
        self.log('train_accu_mean', logs['accu'], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config_training['learning_rate'], betas=(0.9, 0.999), eps=1e-06, weight_decay=self.config_training['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.config_training['learning_rate'], pct_start=self.config_training['warmup'] / self.config_training['num_train_steps'], anneal_strategy=self.config_training['lr_decay'], total_steps=self.config_training['num_train_steps'])
        return ([optimizer], [lr_scheduler])

    def eval_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> PLOutput:
        outputs = self(**batch)
        return outputs

    def eval_epoch_end(self, outputs, prefix: str='train'):
        logs = {}
        counts = torch.tensor([x['count'] for x in outputs]).float()
        logs['count'] = counts.sum()
        for k in ('accu', 'loss'):
            logs[k] = (torch.tensor([x[k] for x in outputs]) * counts).sum() / logs['count']
            self.log(f'{prefix}_{k}_mean', logs[k], sync_dist=True)
        return logs

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> PLOutput:
        outputs = self.eval_step(batch, batch_idx)
        self.logger.log_metrics({f'val_{k}': v for k, v in outputs.items()})
        self.log('val_accu', outputs['accu'], sync_dist=True, prog_bar=True)
        return outputs

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, prefix='val')

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> PLOutput:
        return self.eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, prefix='test')