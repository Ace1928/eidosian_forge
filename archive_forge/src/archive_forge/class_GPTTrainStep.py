import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
class GPTTrainStep(AbstractTrainStep):
    """
    GPT train step class.

    Args:
        args (`argparse.Namespace`): Megatron-LM arguments.
    """

    def __init__(self, args):
        super().__init__('GPTTrainStep')
        self.get_batch = self.get_batch_func(args.megatron_dataset_flag)
        self.loss_func = self.get_loss_func()
        self.forward_step = self.get_forward_step_func()
        self.eod_token = args.padded_vocab_size - 1
        if args.vocab_file is not None:
            tokenizer = get_tokenizer()
            self.eod_token = tokenizer.eod
        self.reset_position_ids = args.reset_position_ids
        self.reset_attention_mask = args.reset_attention_mask
        self.eod_mask_loss = args.eod_mask_loss
        if not args.model_return_dict:
            self.model_output_class = None
        else:
            self.model_output_class = CausalLMOutputWithCrossAttentions

    def get_batch_func(self, megatron_dataset_flag):

        def get_batch_megatron(data_iterator):
            """Generate a batch"""
            keys = ['text']
            datatype = torch.int64
            if data_iterator is not None:
                data = next(data_iterator)
            else:
                data = None
            data_b = mpu.broadcast_data(keys, data, datatype)
            tokens_ = data_b['text'].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, self.eod_mask_loss)
            return (tokens, labels, loss_mask, attention_mask, position_ids)

        def get_batch_transformer(data_iterator):
            data = next(data_iterator)
            data = {'input_ids': data['input_ids']}
            data = send_to_device(data, torch.cuda.current_device())
            tokens_ = data['input_ids'].long()
            padding = torch.zeros((tokens_.shape[0], 1), dtype=tokens_.dtype, device=tokens_.device) + self.eod_token
            tokens_ = torch.concat([tokens_, padding], dim=1)
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens, self.eod_token, self.reset_position_ids, self.reset_attention_mask, True)
            return (tokens, labels, loss_mask, attention_mask, position_ids)
        if megatron_dataset_flag:
            return get_batch_megatron
        else:
            return get_batch_transformer

    def get_loss_func(self):
        args = get_args()

        def loss_func(loss_mask, output_tensor):
            if args.return_logits:
                losses, logits = output_tensor
            else:
                losses = output_tensor
            losses = losses.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
            averaged_loss = average_losses_across_data_parallel_group([loss])
            output_dict = {'lm loss': averaged_loss[0]}
            if args.return_logits:
                output_dict.update({'logits': logits})
            return (loss, output_dict)
        return loss_func

    def get_forward_step_func(self):

        def forward_step(data_iterator, model):
            """Forward step."""
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            return (output_tensor, partial(self.loss_func, loss_mask))
        return forward_step