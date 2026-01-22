from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from uform.torch_models import VisualEncoder
class VLMProcessor(ProcessorMixin):

    def __init__(self, config, **kwargs):
        self.feature_extractor = None
        self.config = config
        if config.center_crop:
            self.image_processor = Compose([Resize(256, interpolation=InterpolationMode.BICUBIC), CenterCrop(config.image_size), convert_to_rgb, ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        else:
            self.image_processor = Compose([RandomResizedCrop(config.image_size, scale=(0.8, 1), interpolation=InterpolationMode.BICUBIC), convert_to_rgb, ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path, additional_special_tokens=['<|im_end|>'])
        self.num_image_latents = config.image_pooler_num_latents

    def __call__(self, texts=None, images=None, return_tensors='pt', **kwargs):
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            tokenized_texts = []
            for text in texts:
                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': f' <image> {text}'}]
                tokenized_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=return_tensors)
                tokenized_texts.append(tokenized_prompt)
            max_len = max((len(t[0]) for t in tokenized_texts))
            input_ids = torch.full((len(tokenized_texts), max_len), fill_value=self.tokenizer.pad_token_id, dtype=torch.int64)
            attention_mask = torch.full((len(tokenized_texts), max_len), fill_value=0, dtype=torch.int64)
            for i, tokens in enumerate(tokenized_texts):
                input_ids[i, -len(tokens[0]):] = tokens[0]
                attention_mask[i, -len(tokens[0]):] = 1
            attention_mask = F.pad(attention_mask, pad=(0, self.num_image_latents - 1), value=1)
            encoding = BatchEncoding(data={'input_ids': input_ids, 'attention_mask': attention_mask})
        if images is not None:
            if isinstance(images, (list, tuple)):
                image_features = torch.empty((len(images), 3, self.config.image_size, self.config.image_size), dtype=torch.float32)
                for i, image in enumerate(images):
                    image_features[i] = self.image_processor(image)
            else:
                image_features = self.image_processor(images).unsqueeze(0)
        if texts is not None and images is not None:
            encoding['images'] = image_features
            return encoding
        if texts is not None:
            return encoding
        return BatchEncoding(data={'images': image_features}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, force_download: bool=False, local_files_only: bool=False, token=None, revision: str='main', **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config)