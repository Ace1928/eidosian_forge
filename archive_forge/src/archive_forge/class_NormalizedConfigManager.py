import functools
from typing import Callable, Dict, Type, Union
from transformers import PretrainedConfig
class NormalizedConfigManager:
    """
    A class that contains all the information needed by ONNX Runtime optimization for a given model type.
    Attributes:
        _conf (`Dict[str, tuple]`):
            A dictionary mapping each supported model type to a tuple containing the number of attention heads
            and the hidden size model config attribute names as well as the corresponding ONNX Runtime model type.
    """
    "\n    TODO: missing normalized configs (currently not useful)\n        ['beit',\n        'clip',\n        'convbert',\n        'convnext',\n        'convnextv2',\n        'data2vec-text',\n        'data2vec-vision',\n        'detr',\n        'flaubert',\n        'groupvit',\n        'ibert',\n        'layoutlm',\n        'layoutlmv3',\n        'levit',\n        'mobilebert',\n        'mobilevit',\n        'owlvit',\n        'perceiver',\n        'roformer',\n        'segformer',\n        'squeezebert',\n        'table-transformer',\n    "
    _conf = {'albert': NormalizedTextConfig, 'bart': BartLikeNormalizedTextConfig, 'bert': NormalizedTextConfig, 'blenderbot': BartLikeNormalizedTextConfig, 'blenderbot-small': BartLikeNormalizedTextConfig, 'bloom': NormalizedTextConfig.with_args(num_layers='n_layer'), 'falcon': NormalizedTextConfig, 'camembert': NormalizedTextConfig, 'codegen': GPT2LikeNormalizedTextConfig, 'cvt': NormalizedVisionConfig, 'deberta': NormalizedTextConfig, 'deberta-v2': NormalizedTextConfig, 'deit': NormalizedVisionConfig, 'distilbert': NormalizedTextConfig.with_args(num_attention_heads='n_heads', hidden_size='dim'), 'donut-swin': NormalizedVisionConfig, 'electra': NormalizedTextConfig, 'encoder-decoder': NormalizedEncoderDecoderConfig, 'gpt2': GPT2LikeNormalizedTextConfig, 'gpt-bigcode': GPTBigCodeNormalizedTextConfig, 'gpt-neo': NormalizedTextConfig.with_args(num_attention_heads='num_heads'), 'gpt-neox': NormalizedTextConfig, 'llama': NormalizedTextConfigWithGQA, 'gptj': GPT2LikeNormalizedTextConfig, 'imagegpt': GPT2LikeNormalizedTextConfig, 'longt5': T5LikeNormalizedTextConfig, 'marian': BartLikeNormalizedTextConfig, 'mbart': BartLikeNormalizedTextConfig, 'mistral': NormalizedTextConfigWithGQA, 'mixtral': NormalizedTextConfigWithGQA, 'mt5': T5LikeNormalizedTextConfig, 'm2m-100': BartLikeNormalizedTextConfig, 'nystromformer': NormalizedTextConfig, 'opt': NormalizedTextConfig, 'pegasus': BartLikeNormalizedTextConfig, 'pix2struct': Pix2StructNormalizedTextConfig, 'phi': NormalizedTextConfig, 'poolformer': NormalizedVisionConfig, 'regnet': NormalizedVisionConfig, 'resnet': NormalizedVisionConfig, 'roberta': NormalizedTextConfig, 'speech-to-text': SpeechToTextLikeNormalizedTextConfig, 'splinter': NormalizedTextConfig, 't5': T5LikeNormalizedTextConfig, 'trocr': TrOCRLikeNormalizedTextConfig, 'whisper': WhisperLikeNormalizedTextConfig, 'vision-encoder-decoder': NormalizedEncoderDecoderConfig, 'vit': NormalizedVisionConfig, 'xlm-roberta': NormalizedTextConfig, 'yolos': NormalizedVisionConfig, 'mpt': MPTNormalizedTextConfig}

    @classmethod
    def check_supported_model(cls, model_type: str):
        if model_type not in cls._conf:
            model_types = ', '.join(cls._conf.keys())
            raise KeyError(f'{model_type} model type is not supported yet in NormalizedConfig. Only {model_types} are supported. If you want to support {model_type} please propose a PR or open up an issue.')

    @classmethod
    def get_normalized_config_class(cls, model_type: str) -> Type:
        model_type = model_type.replace('_', '-')
        cls.check_supported_model(model_type)
        return cls._conf[model_type]