from typing import List
from ...utils.normalized_config import NormalizedConfigManager
from .base import QuantizationApproach
from .config import TextEncoderTFliteConfig, VisionTFLiteConfig
class XLMRobertaTFLiteConfig(DistilBertTFLiteConfig):
    SUPPORTED_QUANTIZATION_APPROACHES = {'default': BertTFLiteConfig.SUPPORTED_QUANTIZATION_APPROACHES, 'question-answering': (QuantizationApproach.INT8_DYNAMIC, QuantizationApproach.FP16)}