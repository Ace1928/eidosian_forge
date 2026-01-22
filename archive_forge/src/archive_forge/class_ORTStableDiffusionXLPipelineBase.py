import importlib
import logging
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from diffusers import (
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer
from transformers.file_utils import add_end_docstrings
import onnxruntime as ort
from ..exporters.onnx import main_export
from ..onnx.utils import _get_external_data_paths
from ..pipelines.diffusers.pipeline_latent_consistency import LatentConsistencyPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipelineMixin
from ..pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from ..pipelines.diffusers.pipeline_utils import VaeImageProcessor
from ..utils import (
from .modeling_ort import ONNX_MODEL_END_DOCSTRING, ORTModel
from .utils import (
class ORTStableDiffusionXLPipelineBase(ORTStableDiffusionPipelineBase):
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def __init__(self, vae_decoder_session: ort.InferenceSession, text_encoder_session: ort.InferenceSession, unet_session: ort.InferenceSession, config: Dict[str, Any], tokenizer: CLIPTokenizer, scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler], feature_extractor: Optional[CLIPFeatureExtractor]=None, vae_encoder_session: Optional[ort.InferenceSession]=None, text_encoder_2_session: Optional[ort.InferenceSession]=None, tokenizer_2: Optional[CLIPTokenizer]=None, use_io_binding: Optional[bool]=None, model_save_dir: Optional[Union[str, Path, TemporaryDirectory]]=None, add_watermarker: Optional[bool]=None):
        super().__init__(vae_decoder_session=vae_decoder_session, text_encoder_session=text_encoder_session, unet_session=unet_session, config=config, tokenizer=tokenizer, scheduler=scheduler, feature_extractor=feature_extractor, vae_encoder_session=vae_encoder_session, text_encoder_2_session=text_encoder_2_session, tokenizer_2=tokenizer_2, use_io_binding=use_io_binding, model_save_dir=model_save_dir)
        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()
        if add_watermarker:
            if not is_invisible_watermark_available():
                raise ImportError('`add_watermarker` requires invisible-watermark to be installed, which can be installed with `pip install invisible-watermark`.')
            from ..pipelines.diffusers.watermark import StableDiffusionXLWatermarker
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None