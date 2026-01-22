import json
import os
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import QuantizationMethod
from ..utils import is_accelerate_available, is_auto_gptq_available
from ..utils.modeling_utils import recurse_getattr
from .constants import GPTQ_CONFIG
from .data import get_dataset, prepare_dataset
from .utils import get_block_name_with_pattern, get_device, get_layers, get_preceding_modules, get_seqlen
class GPTQQuantizer(object):
    """
    A simple API for GPTQ Quantization
    """

    def __init__(self, bits: int, dataset: Optional[Union[List[str], str]]=None, group_size: int=128, damp_percent: float=0.1, desc_act: bool=False, sym: bool=True, true_sequential: bool=True, use_cuda_fp16: bool=False, model_seqlen: Optional[int]=None, block_name_to_quantize: Optional[str]=None, module_name_preceding_first_block: Optional[List[str]]=None, batch_size: int=1, pad_token_id: Optional[int]=None, disable_exllama: bool=False, exllama_config: Dict[str, Any]=None, max_input_length: Optional[int]=None, cache_block_outputs: Optional[bool]=True, modules_in_block_to_quantize: Optional[List[List[str]]]=None, *args, **kwargs):
        """
        Args:
            bits (`int`):
                The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
            dataset (`Union[List[str], str, Any]`, defaults to `None`):
                The dataset used for quantization. You can provide your own dataset in a list of string or in a list of tokenized data
                (e.g. [{ "input_ids": [ 1, 100, 15, ... ],"attention_mask": [ 1, 1, 1, ... ]},...])
                or just use the original datasets used in GPTQ paper ['wikitext2','c4','c4-new','ptb','ptb-new'].
            group_size (int, defaults to 128):
                The group size to use for quantization. Recommended value is 128 and -1 uses per-column quantization.
            damp_percent (`float`, defaults to `0.1`):
                The percent of the average Hessian diagonal to use for dampening, recommended value is 0.1.
            desc_act (`bool`, defaults to `False`):
                Whether to quantize columns in order of decreasing activation size.
                Setting it to False can significantly speed up inference but the perplexity may become slightly worse.
                Also known as act-order.
            sym (`bool`, defaults to `True`):
                Whether to use symetric quantization.
            true_sequential (`bool`, defaults to `True`):
                Whether to perform sequential quantization even within a single Transformer block.
                Instead of quantizing the entire block at once, we perform layer-wise quantization.
                As a result, each layer undergoes quantization using inputs that have passed through the previously quantized layers.
            use_cuda_fp16 (`bool`, defaults to `False`):
                Whether or not to use optimized cuda kernel for fp16 model. Need to have model in fp16.
            model_seqlen (`Optional[int]`, defaults to `None`):
                The maximum sequence length that the model can take.
            block_name_to_quantize (`Optional[str]`, defaults to `None`):
                The transformers block name to quantize. If None, we will infer the block name using common patterns (e.g. model.layers)
            module_name_preceding_first_block (`Optional[List[str]]`, defaults to `None`):
                The layers that are preceding the first Transformer block.
            batch_size (`int`, defaults to `1`):
                The batch size of the dataset
            pad_token_id (`Optional[int]`, defaults to `None`):
                The pad token id. Needed to prepare the dataset when `batch_size` > 1.
            disable_exllama (`bool`, defaults to `False`):
                Whether to use exllama backend. Only works with `bits` = 4.
            exllama_config (`Dict[str, Any]`, *optional*):
                The exllama config. You can specify the version of the exllama kernel through the `version` key. Defaults to `{"version": 2}` if unset.
            max_input_length (`Optional[int]`, defaults to `None`):
                The maximum input length. This is needed to initialize a buffer that depends on the maximum expected input length.
                It is specific to the exllama backend with act-order.
            cache_block_outputs (`bool`, defaults to `True`):
                Whether to cache block outputs to reuse as inputs for the succeeding block. It allows optimization of non-standard models
                (e.g. ChatGLM) but can require more time.
            modules_in_block_to_quantize (`Optional[List[List[str]]]`, defaults to `None`):
                List list of module names to quantize in the block specified. This argument is useful to exclude certain linear modules from being quantized.
                The block to quantize can be specified by setting `block_name_to_quantize`. We will quantize each list sequentially.
                If not set, we will quantize all linear layers. Example: `inside_layer_modules=[["self_attention.query_key_value"], ["mlp.dense_h_to_4h"]]`
        """
        self.bits = bits
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
        self.disable_exllama = disable_exllama
        self.exllama_config = exllama_config
        self.max_input_length = max_input_length
        self.quant_method = QuantizationMethod.GPTQ
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.serialization_keys = ['bits', 'dataset', 'group_size', 'damp_percent', 'desc_act', 'sym', 'true_sequential', 'quant_method', 'modules_in_block_to_quantize']
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError('only support quantize to [2,3,4,8] bits.')
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError('group_size must be greater than 0 or equal to -1')
        if not 0 < self.damp_percent < 1:
            raise ValueError('damp_percent must between 0 and 1.')
        if self.exllama_config is None:
            self.exllama_config = {'version': ExllamaVersion.TWO}
        elif 'version' not in self.exllama_config:
            raise ValueError('`exllama_config` needs to have a `version` key')
        elif self.exllama_config['version'] not in [ExllamaVersion.ONE, ExllamaVersion.TWO]:
            version = self.exllama_config['version']
            raise ValueError(f'Only supported versions are in [ExllamaVersion.ONE, ExllamaVersion.TWO] - not recognized version {version}')
        self.exllama_version = self.exllama_config['version']

    def to_dict(self):
        """
        Returns the args in dict format.
        """
        gptq_dict = {}
        for key in self.serialization_keys:
            gptq_dict[key] = getattr(self, key)
        return gptq_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiates a `GPTQQuantizer` using config_dict as kwargs

        Args:
            config_dict (`Dict[str,Any]`):
                quantization config

        Returns:
            `GPTQQuantizer`:  The quantizer object instantiated from those parameters.
        """
        return cls(**config_dict)

    def convert_model(self, model: nn.Module):
        """
        Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted

        """
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name)
        if self.modules_in_block_to_quantize is not None:
            layers_to_keep = sum(self.modules_in_block_to_quantize, [])
            for name in list(layers_to_be_replaced.keys()):
                if not any((name.endswith(layer) for layer in layers_to_keep)):
                    logger.info(f'Quantization disabled for {name} (only modules_in_block_to_quantize={self.modules_in_block_to_quantize} are quantized)')
                    del layers_to_be_replaced[name]
        self._replace_by_quant_layers(model, layers_to_be_replaced)
        return model

    def get_no_split_module_classes(self, model):
        """
        Get the modules that should not be split across multiple devices.
        Args:
            model (`nn.Module`):
                The input model
        """
        block_class_name = recurse_getattr(model, self.block_name_to_quantize)[0].__class__.__name__
        no_split_module_classes = [block_class_name]
        return no_split_module_classes

    def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str=''):
        """
        Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=self.desc_act, group_size=self.group_size, bits=self.bits, disable_exllama=self.disable_exllama or self.exllama_version != ExllamaVersion.ONE, disable_exllamav2=self.disable_exllama or self.exllama_version != ExllamaVersion.TWO)
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + '.' + attr if name != '' else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                if not self.desc_act or self.group_size == -1:
                    new_layer = QuantLinear(self.bits, self.group_size, in_features, out_features, True, use_cuda_fp16=self.use_cuda_fp16, weight_dtype=layer.weight.dtype)
                else:
                    new_layer = QuantLinear(self.bits, self.group_size, in_features, out_features, True, weight_dtype=layer.weight.dtype)
                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + '.' + name1 if name != '' else name1)

    @torch.no_grad()
    def quantize_model(self, model: nn.Module, tokenizer: Optional[Any]=None):
        """
        Quantizes the model using the dataset

        Args:
            model (`nn.Module`):
                The model to quantize
            tokenizer (Optional[`Any`], defaults to `None`):
                The tokenizer to use in order to prepare the dataset. You can pass either:
                    - A custom tokenizer object.
                    - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                      using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
        Returns:
            `nn.Module`: The quantized model
        """
        if not is_auto_gptq_available():
            raise RuntimeError('auto-gptq is required in order to perform quantzation : `pip install auto-gptq`')
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU found. A GPU is needed to quantize model.')
        model.eval()
        has_config = False
        has_device_map = False
        if hasattr(model, 'config'):
            has_config = True
            use_cache = model.config.use_cache
            model.config.use_cache = False
        if hasattr(model, 'hf_device_map'):
            devices = list(model.hf_device_map.values())
            has_device_map = True
            if 'disk' in devices:
                raise ValueError('disk offload is not supported with GPTQ quantization')
            if 'cpu' in devices or torch.device('cpu') in devices:
                if len(model.hf_device_map) > 1:
                    logger.info('Cpu offload is not recommended. There might be some issues with the memory')
                    hook = None
                    for name, device in model.hf_device_map.items():
                        if device == 'cpu':
                            module = recurse_getattr(model, name)
                            remove_hook_from_module(module, recurse=True)
                            module, hook = cpu_offload_with_hook(module, prev_module_hook=hook)
                else:
                    has_device_map = False
        if hasattr(model, 'dtype'):
            self.use_cuda_fp16 = model.dtype == torch.float16
        if self.model_seqlen is None:
            self.model_seqlen = get_seqlen(model)
        device = get_device(model)
        if isinstance(self.dataset, list) and (not isinstance(self.dataset[0], str)):
            dataset = self.dataset
            logger.info('GPTQQuantizer dataset appears to be already tokenized. Skipping tokenization.')
        else:
            if isinstance(tokenizer, str):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                except Exception:
                    raise ValueError(f'We were not able to get the tokenizer using `AutoTokenizer.from_pretrained`\n                        with the string that you have passed {tokenizer}. If you have a custom tokenizer, you can pass it as input.\n                        For now, we only support quantization for text model. Support for vision, speech and multimodel will come later.')
            if self.dataset is None:
                raise ValueError('You need to pass `dataset` in order to quantize your model')
            elif isinstance(self.dataset, str):
                dataset = get_dataset(self.dataset, tokenizer, seqlen=self.model_seqlen, split='train')
            elif isinstance(self.dataset, list):
                dataset = [tokenizer(data, return_tensors='pt') for data in self.dataset]
            else:
                raise ValueError(f'You need to pass a list of string, a list of tokenized data or a string for `dataset`. Found: {type(self.dataset)}.')
        dataset = prepare_dataset(dataset, pad_token_id=self.pad_token_id, batch_size=self.batch_size)
        layer_inputs = []
        layer_outputs = []
        layer_input_kwargs = []
        if self.block_name_to_quantize is None:
            self.block_name_to_quantize = get_block_name_with_pattern(model)
        if self.module_name_preceding_first_block is None:
            self.module_name_preceding_first_block = get_preceding_modules(model, self.block_name_to_quantize)
        blocks = recurse_getattr(model, self.block_name_to_quantize)
        if not has_device_map:
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f'Module {module_name} was not found in model')
                module = module.to(0)
            blocks[0] = blocks[0].to(0)

        def store_input_hook(_, input, *args):
            kwargs = args[0]
            if input is None:
                if 'hidden_states' in kwargs:
                    input = (kwargs['hidden_states'],)
                else:
                    raise ValueError('No input value found in the foward pass')
            layer_inputs.append(input)
            other_kwargs = {}
            for k, v in kwargs.items():
                if k not in ['hidden_states']:
                    other_kwargs[k] = v
            layer_input_kwargs.append(other_kwargs)
            raise ValueError
        if self.cache_block_outputs:
            handle = blocks[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
            for data in dataset:
                for k, v in data.items():
                    data[k] = v.to(0)
                try:
                    model(**data)
                except ValueError:
                    pass
            handle.remove()
        if not has_device_map:
            blocks[0].to(device)
            for module_name in self.module_name_preceding_first_block:
                module = recurse_getattr(model, module_name)
                if module is None:
                    raise ValueError(f'Module {module_name} was not found in model')
        torch.cuda.empty_cache()
        quantizers = {}
        for i, block in enumerate(tqdm(blocks, desc=f'Quantizing {self.block_name_to_quantize} blocks ')):
            logger.info(f'Start quantizing block {self.block_name_to_quantize} {i + 1}/{len(blocks)}')
            if not self.cache_block_outputs:
                handle = block.register_forward_pre_hook(store_input_hook, with_kwargs=True)
                for data in dataset:
                    for k, v in data.items():
                        data[k] = v.to(0)
                    try:
                        model(**data)
                    except ValueError:
                        pass
                handle.remove()
            if not has_device_map or get_device(block) == torch.device('cpu'):
                block = block.to(0)
            layers = get_layers(block)
            if isinstance(self.modules_in_block_to_quantize, list) and len(self.modules_in_block_to_quantize) > 0:
                if self.true_sequential:
                    layers_name_list = self.modules_in_block_to_quantize
                else:
                    layers_name_list = [sum(self.modules_in_block_to_quantize, [])]
            elif self.true_sequential:
                layers_name_list = [[key] for key in layers.keys()]
            else:
                layers_name_list = [list(layers.keys())]
            logger.info(f'Module to quantize {layers_name_list}')
            for subset_name_list in tqdm(layers_name_list, leave=False, desc='Quantizing layers inside the block'):
                subset_layers = {name: layers[name] for name in subset_name_list}
                gptq = {}
                handles = []
                for name in subset_layers:
                    gptq[name] = GPTQ(subset_layers[name])
                    gptq[name].quantizer.configure(bits=self.bits, sym=self.sym, perchannel=True)

                    def add_batch(name):

                        def tmp(_, input, output):
                            gptq[name].add_batch(input[0].data, output.data)
                        return tmp
                    handles.append(subset_layers[name].register_forward_hook(add_batch(name)))
                for j in range(len(dataset)):
                    block(*layer_inputs[j], **layer_input_kwargs[j])
                for h in handles:
                    h.remove()
                for name in subset_name_list:
                    logger.info(f'Quantizing {name} in block {i + 1}/{len(blocks)}...')
                    scale, zero, g_idx = gptq[name].fasterquant(percdamp=self.damp_percent, group_size=self.group_size, actorder=self.desc_act)
                    quantizers[f'{self.block_name_to_quantize}.{i}.{name}'] = (gptq[name].quantizer, scale, zero, g_idx)
                    gptq[name].free()
                del subset_layers
            if self.cache_block_outputs:
                for j in range(len(dataset)):
                    layer_output = block(*layer_inputs[j], **layer_input_kwargs[j])
                    layer_outputs.append(layer_output)
                if not has_device_map:
                    blocks[i] = block.to(device)
                del layers
                del layer_inputs
                layer_inputs, layer_outputs = (layer_outputs, [])
            else:
                del layers
                del layer_inputs
                layer_inputs = []
            torch.cuda.empty_cache()
        if self.bits == 4:
            if device == torch.device('cpu') or (has_device_map and any((d in devices for d in ['cpu', 'disk']))):
                if not self.disable_exllama:
                    logger.warning('Found modules on cpu/disk. Using Exllama/Exllamav2 backend requires all the modules to be on GPU. Setting `disable_exllama=True`')
                    self.disable_exllama = True
            elif self.desc_act and (not self.disable_exllama) and (self.exllama_version == ExllamaVersion.ONE):
                logger.warning('Using Exllama backend with act_order will reorder the weights offline, thus you will not be able to save the model with the right weights.Setting `disable_exllama=True`. You should only use Exllama backend with act_order for inference. ')
                self.disable_exllama = True
            elif not self.disable_exllama and self.exllama_version == ExllamaVersion.TWO:
                logger.warning('Using Exllamav2 backend will reorder the weights offline, thus you will not be able to save the model with the right weights.Setting `disable_exllama=True`. You should only use Exllamav2 backend for inference. ')
                self.disable_exllama = True
        self.pack_model(model=model, quantizers=quantizers)
        model.is_quantized = True
        model.quantization_method = QuantizationMethod.GPTQ
        if has_config:
            model.config.use_cache = use_cache
            model.config.quantization_config = self.to_dict()
        model = self.post_init_model(model)
        torch.cuda.empty_cache()
        return model

    def post_init_model(self, model):
        """
        Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
        if self.bits == 4 and (not self.disable_exllama):
            if get_device(model) == torch.device('cpu') or (hasattr(model, 'hf_device_map') and any((d in model.hf_device_map for d in ['cpu', 'disk']))):
                raise ValueError('Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU.You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object')

        class StoreAttr(object):
            pass
        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = self.desc_act
        model = autogptq_post_init(model, use_act_order=self.desc_act)
        if self.desc_act and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE) and (self.max_input_length is not None):
            model = exllama_set_max_input_length(model, self.max_input_length)
        return model

    def pack_model(self, model: nn.Module, quantizers: Dict[str, Tuple]):
        """
        Pack the model by replacing the layers by quantized layers

        Args:
            model (`nn.Module`):
                The model to pack
            quantizers (`Dict[str,Tuple]`):
                A mapping of the layer name and the data needed to pack the layer
        """
        QuantLinear = dynamically_import_QuantLinear(use_triton=False, desc_act=self.desc_act, group_size=self.group_size, bits=self.bits, disable_exllama=self.disable_exllama or self.exllama_version != ExllamaVersion.ONE, disable_exllamav2=self.disable_exllama or self.exllama_version != ExllamaVersion.TWO)
        logger.info('Packing model...')
        layers = get_layers(model)
        layers = {n: layers[n] for n in quantizers}
        self._replace_by_quant_layers(model, quantizers)
        qlayers = get_layers(model, [QuantLinear])
        for name in qlayers:
            logger.info(name)
            quantizers[name], scale, zero, g_idx = quantizers[name]
            layer_device = qlayers[name].device
            qlayers[name].to('cpu')
            layers[name], scale, zero, g_idx = (layers[name].to('cpu'), scale.to('cpu'), zero.to('cpu'), g_idx.to('cpu'))
            qlayers[name].pack(layers[name], scale, zero, g_idx)
            qlayers[name].to(layer_device)
        logger.info('Model packed.')

    def save(self, model: nn.Module, save_dir: str, max_shard_size: str='10GB', safe_serialization: bool=True):
        """
        Save model state dict and configs

        Args:
            model (`nn.Module`):
                Model to be saved. The model can be wrapped or unwraped.
            save_dir (`str`):
                Directory to which to save. Will be created if it doesn't exist.
            max_shard_size (`str`, defaults to `"10GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>
            safe_serialization (`bool`, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).

        """
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        with open(os.path.join(save_dir, GPTQ_CONFIG), 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)