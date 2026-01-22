import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from packaging import version
from ..utils import is_auto_awq_available, is_torch_available, logging
@dataclass
class GPTQConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        dataset (`Union[List[str]]`, *optional*):
            The dataset used for quantization. You can provide your own dataset in a list of string or just use the
            original datasets used in GPTQ paper ['wikitext2','c4','c4-new','ptb','ptb-new']
        group_size (`int`, *optional*, defaults to 128):
            The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
        damp_percent (`float`, *optional*, defaults to 0.1):
            The percent of the average Hessian diagonal to use for dampening. Recommended value is 0.1.
        desc_act (`bool`, *optional*, defaults to `False`):
            Whether to quantize columns in order of decreasing activation size. Setting it to False can significantly
            speed up inference but the perplexity may become slightly worse. Also known as act-order.
        sym (`bool`, *optional*, defaults to `True`):
            Whether to use symetric quantization.
        true_sequential (`bool`, *optional*, defaults to `True`):
            Whether to perform sequential quantization even within a single Transformer block. Instead of quantizing
            the entire block at once, we perform layer-wise quantization. As a result, each layer undergoes
            quantization using inputs that have passed through the previously quantized layers.
        use_cuda_fp16 (`bool`, *optional*, defaults to `False`):
            Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
        model_seqlen (`int`, *optional*):
            The maximum sequence length that the model can take.
        block_name_to_quantize (`str`, *optional*):
            The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
        module_name_preceding_first_block (`List[str]`, *optional*):
            The layers that are preceding the first Transformer block.
        batch_size (`int`, *optional*, defaults to 1):
            The batch size used when processing the dataset
        pad_token_id (`int`, *optional*):
            The pad token id. Needed to prepare the dataset when `batch_size` > 1.
        use_exllama (`bool`, *optional*):
            Whether to use exllama backend. Defaults to `True` if unset. Only works with `bits` = 4.
        max_input_length (`int`, *optional*):
            The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input
            length. It is specific to the exllama backend with act-order.
        exllama_config (`Dict[str, Any]`, *optional*):
            The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults
            to `{"version": 1}` if unset.
        cache_block_outputs (`bool`, *optional*, defaults to `True`):
            Whether to cache block outputs to reuse as inputs for the succeeding block.
        modules_in_block_to_quantize (`List[List[str]]`, *optional*):
            List of list of module names to quantize in the specified block. This argument is useful to exclude certain linear modules from being quantized.
            The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially. If not set, we will quantize all linear layers.
            Example: `modules_in_block_to_quantize =[["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]`.
            In this example, we will first quantize the q,k,v layers simultaneously since they are independent.
            Then, we will quantize `self_attn.o_proj` layer with the q,k,v layers quantized. This way, we will get
            better results since it reflects the real input `self_attn.o_proj` will get when the model is quantized.
    """

    def __init__(self, bits: int, tokenizer: Any=None, dataset: Optional[Union[List[str], str]]=None, group_size: int=128, damp_percent: float=0.1, desc_act: bool=False, sym: bool=True, true_sequential: bool=True, use_cuda_fp16: bool=False, model_seqlen: Optional[int]=None, block_name_to_quantize: Optional[str]=None, module_name_preceding_first_block: Optional[List[str]]=None, batch_size: int=1, pad_token_id: Optional[int]=None, use_exllama: Optional[bool]=None, max_input_length: Optional[int]=None, exllama_config: Optional[Dict[str, Any]]=None, cache_block_outputs: bool=True, modules_in_block_to_quantize: Optional[List[List[str]]]=None, **kwargs):
        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.sym = sym
        self.true_sequential = true_sequential
        self.use_cuda_fp16 = use_cuda_fp16
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.use_exllama = use_exllama
        self.max_input_length = max_input_length
        self.exllama_config = exllama_config
        self.disable_exllama = kwargs.pop('disable_exllama', None)
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.post_init()

    def get_loading_attributes(self):
        attibutes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes = ['disable_exllama', 'use_exllama', 'exllama_config', 'use_cuda_fp16', 'max_input_length']
        loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        return loading_attibutes_dict

    def post_init(self):
        """
        Safety checker that arguments are correct
        """
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f'Only support quantization to [2,3,4,8] bits but found {self.bits}')
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError('group_size must be greater than 0 or equal to -1')
        if not 0 < self.damp_percent < 1:
            raise ValueError('damp_percent must between 0 and 1.')
        if self.dataset is not None:
            if isinstance(self.dataset, str):
                if self.dataset not in ['wikitext2', 'c4', 'c4-new', 'ptb', 'ptb-new']:
                    raise ValueError(f"You have entered a string value for dataset. You can only choose between\n                        ['wikitext2','c4','c4-new','ptb','ptb-new'], but we found {self.dataset}")
            elif not isinstance(self.dataset, list):
                raise ValueError(f"dataset needs to be either a list of string or a value in\n                    ['wikitext2','c4','c4-new','ptb','ptb-new'], but we found {self.dataset}")
        if self.disable_exllama is None and self.use_exllama is None:
            self.use_exllama = True
        elif self.disable_exllama is not None and self.use_exllama is None:
            logger.warning('Using `disable_exllama` is deprecated and will be removed in version 4.37. Use `use_exllama` instead and specify the version with `exllama_config`.The value of `use_exllama` will be overwritten by `disable_exllama` passed in `GPTQConfig` or stored in your config file.')
            self.use_exllama = not self.disable_exllama
            self.disable_exllama = None
        elif self.disable_exllama is not None and self.use_exllama is not None:
            raise ValueError('Cannot specify both `disable_exllama` and `use_exllama`. Please use just `use_exllama`')
        if self.exllama_config is None:
            self.exllama_config = {'version': ExllamaVersion.ONE}
        elif 'version' not in self.exllama_config:
            raise ValueError('`exllama_config` needs to have a `version` key.')
        elif self.exllama_config['version'] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
            exllama_version = self.exllama_config['version']
            raise ValueError(f'Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {exllama_version}')
        if self.bits == 4 and self.use_exllama:
            if self.exllama_config['version'] == ExllamaVersion.ONE:
                logger.info('You have activated exllama backend. Note that you can get better inference speed using exllamav2 kernel by setting `exllama_config`.')
            elif self.exllama_config['version'] == ExllamaVersion.TWO:
                optimum_version = version.parse(importlib.metadata.version('optimum'))
                autogptq_version = version.parse(importlib.metadata.version('auto_gptq'))
                if optimum_version <= version.parse('1.13.2') or autogptq_version <= version.parse('0.4.2'):
                    raise ValueError(f'You need optimum > 1.13.2 and auto-gptq > 0.4.2 . Make sure to have that version installed - detected version : optimum {optimum_version} and autogptq {autogptq_version}')
        if self.modules_in_block_to_quantize is not None:
            optimum_version = version.parse(importlib.metadata.version('optimum'))
            if optimum_version < version.parse('1.15.0'):
                raise ValueError('You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument, please upgrade `optimum` package to a version superior than 1.15.0 .')

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop('disable_exllama', None)
        return config_dict

    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        """
        quant_dict = self.to_dict()
        quant_dict['disable_exllama'] = not self.use_exllama
        return quant_dict

    @classmethod
    def from_dict_optimum(cls, config_dict):
        """
        Get compatible class with optimum gptq config dict
        """
        if 'disable_exllama' in config_dict:
            config_dict['use_exllama'] = not config_dict['disable_exllama']
            config_dict['disable_exllama'] = None
        config = cls(**config_dict)
        return config