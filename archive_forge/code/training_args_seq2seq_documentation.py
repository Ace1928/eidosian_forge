from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments
from .training_args import ORTTrainingArguments

    Parameters:
        optim (`str` or [`training_args.ORTOptimizerNames`] or [`transformers.training_args.OptimizerNames`], *optional*, defaults to `"adamw_hf"`):
            The optimizer to use, including optimizers in Transformers: adamw_hf, adamw_torch, adamw_apex_fused, or adafactor. And optimizers implemented by ONNX Runtime: adamw_ort_fused.
    