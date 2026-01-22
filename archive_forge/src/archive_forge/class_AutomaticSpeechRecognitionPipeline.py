from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional, Union
import numpy as np
import requests
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import is_torch_available, is_torchaudio_available, logging
from .audio_utils import ffmpeg_read
from .base import ChunkPipeline
class AutomaticSpeechRecognitionPipeline(ChunkPipeline):
    """
    Pipeline that aims at extracting spoken text contained within some audio.

    The input can be either a raw waveform or a audio file. In case of the audio file, ffmpeg should be installed for
    to support multiple audio formats

    Example:

    ```python
    >>> from transformers import pipeline

    >>> transcriber = pipeline(model="openai/whisper-base")
    >>> transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
    {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    Arguments:
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model that will be used by the pipeline to make predictions. This needs to be a model inheriting from
            [`PreTrainedModel`] for PyTorch and [`TFPreTrainedModel`] for TensorFlow.
        feature_extractor ([`SequenceFeatureExtractor`]):
            The feature extractor that will be used by the pipeline to encode waveform for the model.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer that will be used by the pipeline to encode data for the model. This object inherits from
            [`PreTrainedTokenizer`].
        decoder (`pyctcdecode.BeamSearchDecoderCTC`, *optional*):
            [PyCTCDecode's
            BeamSearchDecoderCTC](https://github.com/kensho-technologies/pyctcdecode/blob/2fd33dc37c4111417e08d89ccd23d28e9b308d19/pyctcdecode/decoder.py#L180)
            can be passed for language model boosted decoding. See [`Wav2Vec2ProcessorWithLM`] for more information.
        chunk_length_s (`float`, *optional*, defaults to 0):
            The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default).

            <Tip>

            For more information on how to effectively use `chunk_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        stride_length_s (`float`, *optional*, defaults to `chunk_length_s / 6`):
            The length of stride on the left and right of each chunk. Used only with `chunk_length_s > 0`. This enables
            the model to *see* more context and infer letters better than without this context but the pipeline
            discards the stride bits at the end to make the final reconstitution as perfect as possible.

            <Tip>

            For more information on how to effectively use `stride_length_s`, please have a look at the [ASR chunking
            blog post](https://huggingface.co/blog/asr-chunking).

            </Tip>

        framework (`str`, *optional*):
            The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
            installed. If no framework is specified, will default to the one currently installed. If no framework is
            specified and both frameworks are installed, will default to the framework of the `model`, or to PyTorch if
            no model is provided.
        device (Union[`int`, `torch.device`], *optional*):
            Device ordinal for CPU/GPU supports. Setting this to `None` will leverage CPU, a positive will run the
            model on the associated CUDA device id.
        torch_dtype (Union[`int`, `torch.dtype`], *optional*):
            The data-type (dtype) of the computation. Setting this to `None` will use float32 precision. Set to
            `torch.float16` or `torch.bfloat16` to use half-precision in the respective dtypes.

    """

    def __init__(self, model: 'PreTrainedModel', feature_extractor: Union['SequenceFeatureExtractor', str]=None, tokenizer: Optional[PreTrainedTokenizer]=None, decoder: Optional[Union['BeamSearchDecoderCTC', str]]=None, device: Union[int, 'torch.device']=None, torch_dtype: Optional[Union[str, 'torch.dtype']]=None, **kwargs):
        if model.config.model_type == 'whisper':
            self.type = 'seq2seq_whisper'
        elif model.__class__.__name__ in MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES.values():
            self.type = 'seq2seq'
        elif feature_extractor._processor_class and feature_extractor._processor_class.endswith('WithLM') and (decoder is not None):
            self.decoder = decoder
            self.type = 'ctc_with_lm'
        else:
            self.type = 'ctc'
        super().__init__(model, tokenizer, feature_extractor, device=device, torch_dtype=torch_dtype, **kwargs)

    def __call__(self, inputs: Union[np.ndarray, bytes, str], **kwargs):
        """
        Transcribe the audio sequence(s) given as inputs to text. See the [`AutomaticSpeechRecognitionPipeline`]
        documentation for more information.

        Args:
            inputs (`np.ndarray` or `bytes` or `str` or `dict`):
                The inputs is either :
                    - `str` that is either the filename of a local audio file, or a public URL address to download the
                      audio file. The file will be read at the correct sampling rate to get the waveform using
                      *ffmpeg*. This requires *ffmpeg* to be installed on the system.
                    - `bytes` it is supposed to be the content of an audio file and is interpreted by *ffmpeg* in the
                      same way.
                    - (`np.ndarray` of shape (n, ) of type `np.float32` or `np.float64`)
                        Raw audio at the correct sampling rate (no further check will be done)
                    - `dict` form can be used to pass raw audio sampled at arbitrary `sampling_rate` and let this
                      pipeline do the resampling. The dict must be in the format `{"sampling_rate": int, "raw":
                      np.array}` with optionally a `"stride": (left: int, right: int)` than can ask the pipeline to
                      treat the first `left` samples and last `right` samples to be ignored in decoding (but used at
                      inference to provide more context to the model). Only use `stride` with CTC models.
            return_timestamps (*optional*, `str` or `bool`):
                Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for
                other sequence-to-sequence models.

                For CTC models, timestamps can take one of two formats:
                    - `"char"`: the pipeline will return timestamps along the text for every character in the text. For
                        instance, if you get `[{"text": "h", "timestamp": (0.5, 0.6)}, {"text": "i", "timestamp": (0.7,
                        0.9)}]`, then it means the model predicts that the letter "h" was spoken after `0.5` and before
                        `0.6` seconds.
                    - `"word"`: the pipeline will return timestamps along the text for every word in the text. For
                        instance, if you get `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text": "there", "timestamp":
                        (1.0, 1.5)}]`, then it means the model predicts that the word "hi" was spoken after `0.5` and
                        before `0.9` seconds.

                For the Whisper model, timestamps can take one of two formats:
                    - `"word"`: same as above for word-level CTC timestamps. Word-level timestamps are predicted
                        through the *dynamic-time warping (DTW)* algorithm, an approximation to word-level timestamps
                        by inspecting the cross-attention weights.
                    - `True`: the pipeline will return timestamps along the text for *segments* of words in the text.
                        For instance, if you get `[{"text": " Hi there!", "timestamp": (0.5, 1.5)}]`, then it means the
                        model predicts that the segment "Hi there!" was spoken after `0.5` and before `1.5` seconds.
                        Note that a segment of text refers to a sequence of one or more words, rather than individual
                        words as with word-level timestamps.
            generate_kwargs (`dict`, *optional*):
                The dictionary of ad-hoc parametrization of `generate_config` to be used for the generation call. For a
                complete overview of generate, check the [following
                guide](https://huggingface.co/docs/transformers/en/main_classes/text_generation).
            max_new_tokens (`int`, *optional*):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.

        Return:
            `Dict`: A dictionary with the following keys:
                - **text** (`str`): The recognized text.
                - **chunks** (*optional(, `List[Dict]`)
                    When using `return_timestamps`, the `chunks` will become a list containing all the various text
                    chunks identified by the model, *e.g.* `[{"text": "hi ", "timestamp": (0.5, 0.9)}, {"text":
                    "there", "timestamp": (1.0, 1.5)}]`. The original full text can roughly be recovered by doing
                    `"".join(chunk["text"] for chunk in output["chunks"])`.
        """
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, chunk_length_s=None, stride_length_s=None, ignore_warning=None, decoder_kwargs=None, return_timestamps=None, return_language=None, generate_kwargs=None, max_new_tokens=None):
        preprocess_params = {}
        if chunk_length_s is not None:
            if self.type == 'seq2seq' and (not ignore_warning):
                logger.warning('Using `chunk_length_s` is very experimental with seq2seq models. The results will not necessarily be entirely accurate and will have caveats. More information: https://github.com/huggingface/transformers/pull/20104. Ignore this warning with pipeline(..., ignore_warning=True)')
            preprocess_params['chunk_length_s'] = chunk_length_s
        if stride_length_s is not None:
            preprocess_params['stride_length_s'] = stride_length_s
        forward_params = defaultdict(dict)
        if max_new_tokens is not None:
            forward_params['generate_kwargs']['max_new_tokens'] = max_new_tokens
        if generate_kwargs is not None:
            if max_new_tokens is not None and 'max_new_tokens' in generate_kwargs:
                raise ValueError('`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use only 1 version')
            forward_params['generate_kwargs'].update(generate_kwargs)
        postprocess_params = {}
        if decoder_kwargs is not None:
            postprocess_params['decoder_kwargs'] = decoder_kwargs
        if return_timestamps is not None:
            if self.type == 'seq2seq' and return_timestamps:
                raise ValueError('We cannot return_timestamps yet on non-CTC models apart from Whisper!')
            if self.type == 'ctc_with_lm' and return_timestamps != 'word':
                raise ValueError("CTC with LM can only predict word level timestamps, set `return_timestamps='word'`")
            if self.type == 'ctc' and return_timestamps not in ['char', 'word']:
                raise ValueError("CTC can either predict character level timestamps, or word level timestamps. Set `return_timestamps='char'` or `return_timestamps='word'` as required.")
            if self.type == 'seq2seq_whisper' and return_timestamps == 'char':
                raise ValueError("Whisper cannot return `char` timestamps, only word level or segment level timestamps. Use `return_timestamps='word'` or `return_timestamps=True` respectively.")
            forward_params['return_timestamps'] = return_timestamps
            postprocess_params['return_timestamps'] = return_timestamps
        if return_language is not None:
            if self.type != 'seq2seq_whisper':
                raise ValueError('Only Whisper can return language for now.')
            postprocess_params['return_language'] = return_language
        return (preprocess_params, forward_params, postprocess_params)

    def preprocess(self, inputs, chunk_length_s=0, stride_length_s=None):
        if isinstance(inputs, str):
            if inputs.startswith('http://') or inputs.startswith('https://'):
                inputs = requests.get(inputs).content
            else:
                with open(inputs, 'rb') as f:
                    inputs = f.read()
        if isinstance(inputs, bytes):
            inputs = ffmpeg_read(inputs, self.feature_extractor.sampling_rate)
        stride = None
        extra = {}
        if isinstance(inputs, dict):
            stride = inputs.pop('stride', None)
            if not ('sampling_rate' in inputs and ('raw' in inputs or 'array' in inputs)):
                raise ValueError('When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "raw" key containing the numpy array representing the audio and a "sampling_rate" key, containing the sampling_rate associated with that array')
            _inputs = inputs.pop('raw', None)
            if _inputs is None:
                inputs.pop('path', None)
                _inputs = inputs.pop('array', None)
            in_sampling_rate = inputs.pop('sampling_rate')
            extra = inputs
            inputs = _inputs
            if in_sampling_rate != self.feature_extractor.sampling_rate:
                if is_torchaudio_available():
                    from torchaudio import functional as F
                else:
                    raise ImportError('torchaudio is required to resample audio samples in AutomaticSpeechRecognitionPipeline. The torchaudio package can be installed through: `pip install torchaudio`.')
                inputs = F.resample(torch.from_numpy(inputs), in_sampling_rate, self.feature_extractor.sampling_rate).numpy()
                ratio = self.feature_extractor.sampling_rate / in_sampling_rate
            else:
                ratio = 1
            if stride is not None:
                if stride[0] + stride[1] > inputs.shape[0]:
                    raise ValueError('Stride is too large for input')
                stride = (inputs.shape[0], int(round(stride[0] * ratio)), int(round(stride[1] * ratio)))
        if not isinstance(inputs, np.ndarray):
            raise ValueError(f'We expect a numpy ndarray as input, got `{type(inputs)}`')
        if len(inputs.shape) != 1:
            raise ValueError('We expect a single channel audio input for AutomaticSpeechRecognitionPipeline')
        if chunk_length_s:
            if stride_length_s is None:
                stride_length_s = chunk_length_s / 6
            if isinstance(stride_length_s, (int, float)):
                stride_length_s = [stride_length_s, stride_length_s]
            align_to = getattr(self.model.config, 'inputs_to_logits_ratio', 1)
            chunk_len = int(round(chunk_length_s * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_left = int(round(stride_length_s[0] * self.feature_extractor.sampling_rate / align_to) * align_to)
            stride_right = int(round(stride_length_s[1] * self.feature_extractor.sampling_rate / align_to) * align_to)
            if chunk_len < stride_left + stride_right:
                raise ValueError('Chunk length must be superior to stride length')
            for item in chunk_iter(inputs, self.feature_extractor, chunk_len, stride_left, stride_right, self.torch_dtype):
                yield item
        else:
            if self.type == 'seq2seq_whisper' and inputs.shape[0] > self.feature_extractor.n_samples:
                processed = self.feature_extractor(inputs, sampling_rate=self.feature_extractor.sampling_rate, truncation=False, padding='longest', return_tensors='pt')
            else:
                processed = self.feature_extractor(inputs, sampling_rate=self.feature_extractor.sampling_rate, return_tensors='pt')
            if self.torch_dtype is not None:
                processed = processed.to(dtype=self.torch_dtype)
            if stride is not None:
                if self.type == 'seq2seq':
                    raise ValueError('Stride is only usable with CTC models, try removing it !')
                processed['stride'] = stride
            yield {'is_last': True, **processed, **extra}

    def _forward(self, model_inputs, return_timestamps=False, generate_kwargs=None):
        if generate_kwargs is None:
            generate_kwargs = {}
        attention_mask = model_inputs.pop('attention_mask', None)
        stride = model_inputs.pop('stride', None)
        is_last = model_inputs.pop('is_last')
        if self.type in {'seq2seq', 'seq2seq_whisper'}:
            encoder = self.model.get_encoder()
            if 'input_features' in model_inputs:
                inputs = model_inputs.pop('input_features')
            elif 'input_values' in model_inputs:
                inputs = model_inputs.pop('input_values')
            else:
                raise ValueError(f'Seq2Seq speech recognition model requires either a `input_features` or `input_values` key, but only has {model_inputs.keys()}')
            if return_timestamps and self.type == 'seq2seq_whisper':
                generate_kwargs['return_timestamps'] = return_timestamps
                if return_timestamps == 'word':
                    generate_kwargs['return_token_timestamps'] = True
                    if stride is not None:
                        if isinstance(stride, tuple):
                            generate_kwargs['num_frames'] = stride[0] // self.feature_extractor.hop_length
                        else:
                            generate_kwargs['num_frames'] = [s[0] // self.feature_extractor.hop_length for s in stride]
            if self.type == 'seq2seq_whisper' and inputs.shape[-1] > self.feature_extractor.nb_max_frames:
                generate_kwargs['input_features'] = inputs
            else:
                generate_kwargs['encoder_outputs'] = encoder(inputs, attention_mask=attention_mask)
            tokens = self.model.generate(attention_mask=attention_mask, **generate_kwargs)
            if return_timestamps == 'word' and self.type == 'seq2seq_whisper':
                out = {'tokens': tokens['sequences'], 'token_timestamps': tokens['token_timestamps']}
            else:
                out = {'tokens': tokens}
            if self.type == 'seq2seq_whisper':
                if stride is not None:
                    out['stride'] = stride
        else:
            inputs = {self.model.main_input_name: model_inputs.pop(self.model.main_input_name), 'attention_mask': attention_mask}
            outputs = self.model(**inputs)
            logits = outputs.logits
            if self.type == 'ctc_with_lm':
                out = {'logits': logits}
            else:
                out = {'tokens': logits.argmax(dim=-1)}
            if stride is not None:
                ratio = 1 / self.model.config.inputs_to_logits_ratio
                if isinstance(stride, tuple):
                    out['stride'] = rescale_stride([stride], ratio)[0]
                else:
                    out['stride'] = rescale_stride(stride, ratio)
        extra = model_inputs
        return {'is_last': is_last, **out, **extra}

    def postprocess(self, model_outputs, decoder_kwargs: Optional[Dict]=None, return_timestamps=None, return_language=None):
        optional = {}
        final_items = []
        key = 'logits' if self.type == 'ctc_with_lm' else 'tokens'
        stride = None
        for outputs in model_outputs:
            items = outputs[key].numpy()
            stride = outputs.get('stride', None)
            if stride is not None and self.type in {'ctc', 'ctc_with_lm'}:
                total_n, left, right = stride
                right_n = total_n - right
                items = items[:, left:right_n]
            final_items.append(items)
        if stride and self.type == 'seq2seq':
            items = _find_longest_common_sequence(final_items, self.tokenizer)
        elif self.type == 'seq2seq_whisper':
            time_precision = self.feature_extractor.chunk_length / self.model.config.max_source_positions
            sampling_rate = self.feature_extractor.sampling_rate
            for output in model_outputs:
                if 'stride' in output:
                    chunk_len, stride_left, stride_right = output['stride']
                    chunk_len /= sampling_rate
                    stride_left /= sampling_rate
                    stride_right /= sampling_rate
                    output['stride'] = (chunk_len, stride_left, stride_right)
            text, optional = self.tokenizer._decode_asr(model_outputs, return_timestamps=return_timestamps, return_language=return_language, time_precision=time_precision)
        else:
            items = np.concatenate(final_items, axis=1)
            items = items.squeeze(0)
        if self.type == 'ctc_with_lm':
            if decoder_kwargs is None:
                decoder_kwargs = {}
            beams = self.decoder.decode_beams(items, **decoder_kwargs)
            text = beams[0][0]
            if return_timestamps:
                chunk_offset = beams[0][2]
                offsets = []
                for word, (start_offset, end_offset) in chunk_offset:
                    offsets.append({'word': word, 'start_offset': start_offset, 'end_offset': end_offset})
        elif self.type != 'seq2seq_whisper':
            skip_special_tokens = self.type != 'ctc'
            text = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens)
            if return_timestamps:
                offsets = self.tokenizer.decode(items, skip_special_tokens=skip_special_tokens, output_char_offsets=True)['char_offsets']
                if return_timestamps == 'word':
                    offsets = self.tokenizer._get_word_offsets(offsets, self.tokenizer.replace_word_delimiter_char)
        if return_timestamps and self.type not in {'seq2seq', 'seq2seq_whisper'}:
            chunks = []
            for item in offsets:
                start = item['start_offset'] * self.model.config.inputs_to_logits_ratio
                start /= self.feature_extractor.sampling_rate
                stop = item['end_offset'] * self.model.config.inputs_to_logits_ratio
                stop /= self.feature_extractor.sampling_rate
                chunks.append({'text': item[return_timestamps], 'timestamp': (start, stop)})
            optional['chunks'] = chunks
        extra = defaultdict(list)
        for output in model_outputs:
            output.pop('tokens', None)
            output.pop('logits', None)
            output.pop('is_last', None)
            output.pop('stride', None)
            output.pop('token_timestamps', None)
            for k, v in output.items():
                extra[k].append(v)
        return {'text': text, **optional, **extra}