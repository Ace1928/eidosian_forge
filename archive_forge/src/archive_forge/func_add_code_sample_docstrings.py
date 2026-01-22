import functools
import re
import types
def add_code_sample_docstrings(*docstr, processor_class=None, checkpoint=None, output_type=None, config_class=None, mask='[MASK]', qa_target_start_index=14, qa_target_end_index=15, model_cls=None, modality=None, expected_output=None, expected_loss=None, real_checkpoint=None, revision=None):

    def docstring_decorator(fn):
        model_class = fn.__qualname__.split('.')[0] if model_cls is None else model_cls
        if model_class[:2] == 'TF':
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == 'Flax':
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS
        doc_kwargs = {'model_class': model_class, 'processor_class': processor_class, 'checkpoint': checkpoint, 'mask': mask, 'qa_target_start_index': qa_target_start_index, 'qa_target_end_index': qa_target_end_index, 'expected_output': expected_output, 'expected_loss': expected_loss, 'real_checkpoint': real_checkpoint, 'fake_checkpoint': checkpoint, 'true': '{true}'}
        if ('SequenceClassification' in model_class or 'AudioClassification' in model_class) and modality == 'audio':
            code_sample = sample_docstrings['AudioClassification']
        elif 'SequenceClassification' in model_class:
            code_sample = sample_docstrings['SequenceClassification']
        elif 'QuestionAnswering' in model_class:
            code_sample = sample_docstrings['QuestionAnswering']
        elif 'TokenClassification' in model_class:
            code_sample = sample_docstrings['TokenClassification']
        elif 'MultipleChoice' in model_class:
            code_sample = sample_docstrings['MultipleChoice']
        elif 'MaskedLM' in model_class or model_class in ['FlaubertWithLMHeadModel', 'XLMWithLMHeadModel']:
            code_sample = sample_docstrings['MaskedLM']
        elif 'LMHead' in model_class or 'CausalLM' in model_class:
            code_sample = sample_docstrings['LMHead']
        elif 'CTC' in model_class:
            code_sample = sample_docstrings['CTC']
        elif 'AudioFrameClassification' in model_class:
            code_sample = sample_docstrings['AudioFrameClassification']
        elif 'XVector' in model_class and modality == 'audio':
            code_sample = sample_docstrings['AudioXVector']
        elif 'Model' in model_class and modality == 'audio':
            code_sample = sample_docstrings['SpeechBaseModel']
        elif 'Model' in model_class and modality == 'vision':
            code_sample = sample_docstrings['VisionBaseModel']
        elif 'Model' in model_class or 'Encoder' in model_class:
            code_sample = sample_docstrings['BaseModel']
        elif 'ImageClassification' in model_class:
            code_sample = sample_docstrings['ImageClassification']
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")
        code_sample = filter_outputs_from_example(code_sample, expected_output=expected_output, expected_loss=expected_loss)
        if real_checkpoint is not None:
            code_sample = FAKE_MODEL_DISCLAIMER + code_sample
        func_doc = (fn.__doc__ or '') + ''.join(docstr)
        output_doc = '' if output_type is None else _prepare_output_docstrings(output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)
        if revision is not None:
            if re.match('^refs/pr/\\\\d+', revision):
                raise ValueError(f"The provided revision '{revision}' is incorrect. It should point to a pull request reference on the hub like 'refs/pr/6'")
            built_doc = built_doc.replace(f'from_pretrained("{checkpoint}")', f'from_pretrained("{checkpoint}", revision="{revision}")')
        fn.__doc__ = func_doc + output_doc + built_doc
        return fn
    return docstring_decorator