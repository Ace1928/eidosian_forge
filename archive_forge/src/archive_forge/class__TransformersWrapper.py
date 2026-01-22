from __future__ import annotations
import ast
import base64
import binascii
import contextlib
import copy
import functools
import importlib
import json
import logging
import os
import pathlib
import re
import shutil
import string
import sys
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import yaml
from packaging.version import Version
from mlflow import pyfunc
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import (
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.protos.databricks_pb2 import (
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _get_root_uri_and_artifact_path
from mlflow.transformers.flavor_config import (
from mlflow.transformers.hub_utils import is_valid_hf_repo_id
from mlflow.transformers.llm_inference_utils import (
from mlflow.transformers.model_io import (
from mlflow.transformers.peft import (
from mlflow.transformers.signature import (
from mlflow.transformers.torch_utils import _TORCH_DTYPE_KEY, _deserialize_torch_dtype
from mlflow.types.utils import _validate_input_dictionary_contains_only_strings_and_lists_of_strings
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import (
from mlflow.utils.environment import (
from mlflow.utils.file_utils import TempDir, get_total_file_size, write_to
from mlflow.utils.logging_utils import suppress_logs
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
class _TransformersWrapper:

    def __init__(self, pipeline, flavor_config=None, model_config=None, prompt_template=None):
        self.pipeline = pipeline
        self.flavor_config = flavor_config
        self.model_config = MappingProxyType(model_config or {})
        self.prompt_template = prompt_template
        self._conversation = None
        self._supported_custom_generator_types = {'InstructionTextGenerationPipeline'}
        self.llm_inference_task = self.flavor_config.get(_LLM_INFERENCE_TASK_KEY) if self.flavor_config else None

    def _convert_pandas_to_dict(self, data):
        import transformers
        if not isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            return data.to_dict(orient='records')
        else:
            unpacked = data.to_dict(orient='list')
            parsed = {}
            for key, value in unpacked.items():
                if isinstance(value, list):
                    contents = []
                    for item in value:
                        if item not in contents:
                            contents.append(item)
                    parsed[key] = contents if all((isinstance(item, str) for item in contents)) and len(contents) > 1 else contents[0]
            return parsed

    def _merge_model_config_with_params(self, model_config, params):
        if params:
            _logger.warning('params provided to the `predict` method will override the inference configuration saved with the model. If the params provided are not valid for the pipeline, MlflowException will be raised.')
            return {**model_config, **params}
        else:
            return model_config

    def _validate_model_config_and_return_output(self, data, model_config, return_tensors=False):
        import transformers
        if return_tensors:
            model_config['return_tensors'] = True
            if model_config.get('return_full_text', None) is not None:
                _logger.warning('The `return_full_text` parameter is mutually exclusive with the `return_tensors` parameter set when a MLflow inference task is provided. The `return_full_text` parameter will be ignored.')
                model_config['return_full_text'] = None
        try:
            if isinstance(data, dict):
                return self.pipeline(**data, **model_config)
            return self.pipeline(data, **model_config)
        except ValueError as e:
            if 'The following `model_kwargs` are not used by the model' in str(e):
                raise MlflowException.invalid_parameter_value(f'The params provided to the `predict` method are not valid for pipeline {type(self.pipeline).__name__}.') from e
            if isinstance(self.pipeline, (transformers.AutomaticSpeechRecognitionPipeline, transformers.AudioClassificationPipeline)) and ('Malformed soundfile' in str(e) or 'Soundfile is either not in the correct format or is malformed' in str(e)):
                raise MlflowException.invalid_parameter_value('Failed to process the input audio data. Either the audio file is corrupted or a uri was passed in without overriding the default model signature. If submitting a string uri, please ensure that the model has been saved with a signature that defines a string input type.') from e
            raise

    def predict(self, data, params: Optional[Dict[str, Any]]=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

                .. Note:: Experimental: This parameter may change or be removed in a future
                                        release without warning.

        Returns:
            Model predictions.
        """
        if self.llm_inference_task == _LLM_INFERENCE_TASK_CHAT:
            convert_data_messages_with_chat_template(data, self.pipeline.tokenizer)
        if self.llm_inference_task:
            data, params = preprocess_llm_inference_params(data, self.flavor_config)
        model_config = copy.deepcopy(dict(self.model_config))
        model_config = self._merge_model_config_with_params(model_config, params)
        if isinstance(data, pd.DataFrame):
            input_data = self._convert_pandas_to_dict(data)
        elif isinstance(data, (dict, str, bytes, np.ndarray)):
            input_data = data
        elif isinstance(data, list):
            if not all((isinstance(entry, (str, dict)) for entry in data)):
                raise MlflowException('Invalid data submission. Ensure all elements in the list are strings or dictionaries. If dictionaries are supplied, all keys in the dictionaries must be strings and values must be either str or List[str].', error_code=INVALID_PARAMETER_VALUE)
            input_data = data
        else:
            raise MlflowException('Input data must be either a pandas.DataFrame, a string, bytes, List[str], List[Dict[str, str]], List[Dict[str, Union[str, List[str]]]], or Dict[str, Union[str, List[str]]].', error_code=INVALID_PARAMETER_VALUE)
        input_data = self._parse_raw_pipeline_input(input_data)
        if isinstance(input_data, dict):
            _validate_input_dictionary_contains_only_strings_and_lists_of_strings(input_data)
        elif isinstance(input_data, list) and all((isinstance(entry, dict) for entry in input_data)):
            all((_validate_input_dictionary_contains_only_strings_and_lists_of_strings(x) for x in input_data))
        return self._predict(input_data, model_config)

    def _predict(self, data, model_config):
        import transformers
        if isinstance(self.pipeline, transformers.TranslationPipeline):
            self._validate_str_or_list_str(data)
            output_key = 'translation_text'
        elif isinstance(self.pipeline, transformers.SummarizationPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = 'summary_text'
        elif isinstance(self.pipeline, transformers.Text2TextGenerationPipeline):
            data = self._parse_text2text_input(data)
            data = self._format_prompt_template(data)
            output_key = 'generated_text'
        elif isinstance(self.pipeline, transformers.TextGenerationPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = 'generated_text'
        elif isinstance(self.pipeline, transformers.QuestionAnsweringPipeline):
            data = self._parse_question_answer_input(data)
            output_key = 'answer'
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            self._validate_str_or_list_str(data)
            data = self._format_prompt_template(data)
            output_key = 'token_str'
        elif isinstance(self.pipeline, transformers.TextClassificationPipeline):
            output_key = 'label'
        elif isinstance(self.pipeline, transformers.ImageClassificationPipeline):
            data = self._convert_image_input(data)
            output_key = 'label'
        elif isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            output_key = 'labels'
            data = self._parse_json_encoded_list(data, 'candidate_labels')
        elif isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            output_key = 'answer'
            data = self._parse_json_encoded_dict_payload_to_dict(data, 'table')
        elif isinstance(self.pipeline, transformers.TokenClassificationPipeline):
            output_key = {'entity_group', 'entity'}
        elif isinstance(self.pipeline, transformers.FeatureExtractionPipeline):
            output_key = None
            data = self._parse_feature_extraction_input(data)
            data = self._format_prompt_template(data)
        elif isinstance(self.pipeline, transformers.ConversationalPipeline):
            output_key = None
            if not self._conversation:
                self._conversation = transformers.Conversation()
            self._conversation.add_user_input(data)
        elif type(self.pipeline).__name__ in self._supported_custom_generator_types:
            self._validate_str_or_list_str(data)
            output_key = 'generated_text'
        elif isinstance(self.pipeline, transformers.AutomaticSpeechRecognitionPipeline):
            if model_config.get('return_timestamps', None) in ['word', 'char']:
                output_key = None
            else:
                output_key = 'text'
            data = self._convert_audio_input(data)
        elif isinstance(self.pipeline, transformers.AudioClassificationPipeline):
            data = self._convert_audio_input(data)
            output_key = None
        else:
            raise MlflowException(f'The loaded pipeline type {type(self.pipeline).__name__} is not enabled for pyfunc predict functionality.', error_code=BAD_REQUEST)
        include_prompt = model_config.pop('include_prompt', True)
        collapse_whitespace = model_config.pop('collapse_whitespace', False)
        data = self._convert_cast_lists_from_np_back_to_list(data)
        if isinstance(self.pipeline, transformers.ConversationalPipeline):
            conversation_output = self.pipeline(self._conversation)
            return conversation_output.generated_responses[-1]
        else:
            return_tensors = False
            if self.llm_inference_task:
                return_tensors = True
                output_key = 'generated_token_ids'
            raw_output = self._validate_model_config_and_return_output(data, model_config=model_config, return_tensors=return_tensors)
        if type(self.pipeline).__name__ in self._supported_custom_generator_types or isinstance(self.pipeline, transformers.TextGenerationPipeline):
            output = self._strip_input_from_response_in_instruction_pipelines(data, raw_output, output_key, self.flavor_config, include_prompt, collapse_whitespace)
            if self.llm_inference_task:
                output = postprocess_output_for_llm_inference_task(data, output, self.pipeline, self.flavor_config, model_config, self.llm_inference_task)
        elif isinstance(self.pipeline, transformers.FeatureExtractionPipeline):
            return self._parse_feature_extraction_output(raw_output)
        elif isinstance(self.pipeline, transformers.FillMaskPipeline):
            output = self._parse_list_of_multiple_dicts(raw_output, output_key)
        elif isinstance(self.pipeline, transformers.ZeroShotClassificationPipeline):
            return self._flatten_zero_shot_text_classifier_output_to_df(raw_output)
        elif isinstance(self.pipeline, transformers.TokenClassificationPipeline):
            output = self._parse_tokenizer_output(raw_output, output_key)
        elif isinstance(self.pipeline, transformers.AutomaticSpeechRecognitionPipeline) and model_config.get('return_timestamps', None) in ['word', 'char']:
            output = json.dumps(raw_output)
        elif isinstance(self.pipeline, (transformers.AudioClassificationPipeline, transformers.TextClassificationPipeline, transformers.ImageClassificationPipeline)):
            return pd.DataFrame(raw_output)
        else:
            output = self._parse_lists_of_dict_to_list_of_str(raw_output, output_key)
        sanitized = self._sanitize_output(output, data)
        return self._wrap_strings_as_list_if_scalar(sanitized)

    def _parse_raw_pipeline_input(self, data):
        """
        Converts inputs to the expected types for specific Pipeline types.
        Specific logic for individual pipeline types are called via their respective methods if
        the input isn't a basic str or List[str] input type of Pipeline.
        These parsers are required due to the conversion that occurs within schema validation to
        a Pandas DataFrame encapsulation, a format which is unsupported for the `transformers`
        library.
        """
        import transformers
        data = self._coerce_exploded_dict_to_single_dict(data)
        data = self._parse_input_for_table_question_answering(data)
        data = self._parse_conversation_input(data)
        if isinstance(self.pipeline, (transformers.FillMaskPipeline, transformers.TextGenerationPipeline, transformers.TranslationPipeline, transformers.SummarizationPipeline, transformers.TokenClassificationPipeline)) and isinstance(data, list) and all((isinstance(entry, dict) for entry in data)):
            return [list(entry.values())[0] for entry in data]
        elif isinstance(self.pipeline, transformers.Text2TextGenerationPipeline) and isinstance(data, list) and all((isinstance(entry, dict) for entry in data)) and (0 in data[0].keys()):
            return [list(entry.values())[0] for entry in data]
        elif isinstance(self.pipeline, transformers.TextClassificationPipeline):
            return self._validate_text_classification_input(data)
        else:
            return data

    @staticmethod
    def _validate_text_classification_input(data):
        """
        Perform input type validation for TextClassification pipelines and casting of data
        that is manipulated internally by the MLflow model server back to a structure that
        can be used for pipeline inference.

        To illustrate the input and outputs of this function, for the following inputs to
        the pyfunc.predict() call for this pipeline type:

        "text to classify"
        ["text to classify", "other text to classify"]
        {"text": "text to classify", "text_pair": "pair text"}
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Pyfunc processing will convert these to the following structures:

        [{0: "text to classify"}]
        [{0: "text to classify"}, {0: "other text to classify"}]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        The purpose of this function is to convert them into the correct format for input
        to the pipeline (wrapping as a list has no bearing on the correctness of the
        inferred classifications):

        ["text to classify"]
        ["text to classify", "other text to classify"]
        [{"text": "text to classify", "text_pair": "pair text"}]
        [{"text": "text", "text_pair": "pair"}, {"text": "t", "text_pair": "tp" }]

        Additionally, for dict input types (the 'text' & 'text_pair' input example), the dict
        input will be JSON stringified within MLflow model serving. In order to reconvert this
        structure back into the appropriate type, we use ast.literal_eval() to convert back
        to a dict. We avoid using JSON.loads() due to pandas DataFrame conversions that invert
        single and double quotes with escape sequences that are not consistent if the string
        contains escaped quotes.
        """

        def _check_keys(payload):
            """Check if a dictionary contains only allowable keys."""
            allowable_str_keys = {'text', 'text_pair'}
            if set(payload) - allowable_str_keys and (not all((isinstance(key, int) for key in payload.keys()))):
                raise MlflowException(f'Text Classification pipelines may only define dictionary inputs with keys defined as {allowable_str_keys}')
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            _check_keys(data)
            return data
        elif isinstance(data, list):
            if all((isinstance(item, str) for item in data)):
                return data
            elif all((isinstance(item, dict) for item in data)):
                for payload in data:
                    _check_keys(payload)
                if list(data[0].keys())[0] == 0:
                    data = [item[0] for item in data]
                try:
                    return [ast.literal_eval(s) for s in data]
                except (ValueError, SyntaxError):
                    return data
            else:
                raise MlflowException('An unsupported data type has been passed for Text Classification inference. Only str, list of str, dict, and list of dict are supported.')
        else:
            raise MlflowException('An unsupported data type has been passed for Text Classification inference. Only str, list of str, dict, and list of dict are supported.')

    def _parse_conversation_input(self, data):
        import transformers
        if not isinstance(self.pipeline, transformers.ConversationalPipeline) or isinstance(data, str):
            return data
        elif isinstance(data, list) and all((isinstance(elem, dict) for elem in data)):
            return next(iter(data[0].values()))
        elif isinstance(data, dict):
            return next(iter(data.values()))

    def _parse_input_for_table_question_answering(self, data):
        import transformers
        if not isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            return data
        if 'table' not in data:
            raise MlflowException("The input dictionary must have the 'table' key.", error_code=INVALID_PARAMETER_VALUE)
        elif isinstance(data['table'], dict):
            data['table'] = json.dumps(data['table'])
            return data
        else:
            return data

    def _coerce_exploded_dict_to_single_dict(self, data):
        """
        Parses the result of Pandas DataFrame.to_dict(orient="records") from pyfunc
        signature validation to coerce the output to the required format for a
        Pipeline that requires a single dict with list elements such as
        TableQuestionAnsweringPipeline.
        Example input:

        [
          {"answer": "We should order more pizzas to meet the demand."},
          {"answer": "The venue size should be updated to handle the number of guests."},
        ]

        Output:

        [
          "We should order more pizzas to meet the demand.",
          "The venue size should be updated to handle the number of guests.",
        ]

        """
        import transformers
        if not isinstance(self.pipeline, transformers.TableQuestionAnsweringPipeline):
            return data
        elif isinstance(data, list) and all((isinstance(item, dict) for item in data)):
            collection = data.copy()
            parsed = collection[0]
            for coll in collection:
                for key, value in coll.items():
                    if key not in parsed:
                        raise MlflowException('Unable to parse the input. The keys within each dictionary of the parsed input are not consistentamong the dictionaries.', error_code=INVALID_PARAMETER_VALUE)
                    if value != parsed[key]:
                        value_type = type(parsed[key])
                        if value_type == str:
                            parsed[key] = [parsed[key], value]
                        elif value_type == list:
                            if all((len(entry) == 1 for entry in value)):
                                parsed[key].append([str(value)][0])
                            else:
                                parsed[key] = parsed[key].append(value)
                        else:
                            parsed[key] = value
            return parsed
        else:
            return data

    def _flatten_zero_shot_text_classifier_output_to_df(self, data):
        """
        Converts the output of sequences, labels, and scores to a Pandas DataFrame output.

        Example input:

        [{'sequence': 'My dog loves to eat spaghetti',
          'labels': ['happy', 'sad'],
          'scores': [0.9896970987319946, 0.010302911512553692]},
         {'sequence': 'My dog hates going to the vet',
          'labels': ['sad', 'happy'],
          'scores': [0.957074761390686, 0.042925238609313965]}]

        Output:

        pd.DataFrame in a fully normalized (flattened) format with each sequence, label, and score
        having a row entry.
        For example, here is the DataFrame output:

                                sequence labels    scores
        0  My dog loves to eat spaghetti  happy  0.989697
        1  My dog loves to eat spaghetti    sad  0.010303
        2  My dog hates going to the vet    sad  0.957075
        3  My dog hates going to the vet  happy  0.042925
        """
        if isinstance(data, list) and (not all((isinstance(item, dict) for item in data))):
            raise MlflowException(f'Encountered an unknown return type from the pipeline type {type(self.pipeline).__name__}. Expecting a List[Dict]', error_code=BAD_REQUEST)
        if isinstance(data, dict):
            data = [data]
        flattened_data = []
        for entry in data:
            for label, score in zip(entry['labels'], entry['scores']):
                flattened_data.append({'sequence': entry['sequence'], 'labels': label, 'scores': score})
        return pd.DataFrame(flattened_data)

    def _strip_input_from_response_in_instruction_pipelines(self, input_data, output, output_key, flavor_config, include_prompt=True, collapse_whitespace=False):
        """
        Parse the output from instruction pipelines to conform with other text generator
        pipeline types and remove line feed characters and other confusing outputs
        """

        def extract_response_data(data_out):
            if all((isinstance(x, dict) for x in data_out)):
                return [elem[output_key] for elem in data_out][0]
            elif all((isinstance(x, list) for x in data_out)):
                return [elem[output_key] for coll in data_out for elem in coll]
            else:
                raise MlflowException(f'Unable to parse the pipeline output. Expected List[Dict[str,str]] or List[List[Dict[str,str]]] but got {type(data_out)} instead.')
        output = extract_response_data(output)

        def trim_input(data_in, data_out):
            if not include_prompt and flavor_config[FlavorKey.INSTANCE_TYPE] in self._supported_custom_generator_types and data_out.startswith(data_in + '\n\n'):
                data_out = data_out[len(data_in):].lstrip()
                if data_out.startswith('A:'):
                    data_out = data_out[2:].lstrip()
            if collapse_whitespace:
                data_out = re.sub('\\s+', ' ', data_out).strip()
            return data_out
        if isinstance(input_data, list) and isinstance(output, list):
            return [trim_input(data_in, data_out) for data_in, data_out in zip(input_data, output)]
        elif isinstance(input_data, str) and isinstance(output, str):
            return trim_input(input_data, output)
        else:
            raise MlflowException(f'Unknown data structure after parsing output. Expected str or List[str]. Got {type(output)} instead.')

    def _sanitize_output(self, output, input_data):
        import transformers
        if not isinstance(self.pipeline, transformers.TokenClassificationPipeline) and isinstance(input_data, str) and isinstance(output, list):
            output = output[0]
        if isinstance(output, str):
            return output.strip()
        elif isinstance(output, list):
            if all((isinstance(elem, str) for elem in output)):
                cleaned = [text.strip() for text in output]
                return cleaned if len(cleaned) > 1 else cleaned[0]
            else:
                return [self._sanitize_output(coll, input_data) for coll in output]
        elif isinstance(output, dict) and all((isinstance(key, str) and isinstance(value, str) for key, value in output.items())):
            return {k: v.strip() for k, v in output.items()}
        else:
            return output

    @staticmethod
    def _wrap_strings_as_list_if_scalar(output_data):
        """
        Wraps single string outputs in a list to support batch processing logic in serving.
        Scalar values are not supported for processing in batch logic as they cannot be coerced
        to DataFrame representations.
        """
        if isinstance(output_data, str):
            return [output_data]
        else:
            return output_data

    def _parse_lists_of_dict_to_list_of_str(self, output_data, target_dict_key) -> List[str]:
        """
        Parses the output results from select Pipeline types to extract specific values from a
        target key.
        Examples (with "a" as the `target_dict_key`):

        Input: [{"a": "valid", "b": "invalid"}, {"a": "another valid", "c": invalid"}]
        Output: ["valid", "another_valid"]

        Input: [{"a": "valid", "b": [{"a": "another valid"}, {"b": "invalid"}]},
                {"a": "valid 2", "b": [{"a": "another valid 2"}, {"c": "invalid"}]}]
        Output: ["valid", "another valid", "valid 2", "another valid 2"]
        """
        if isinstance(output_data, list):
            output_coll = []
            for output in output_data:
                if isinstance(output, dict):
                    for key, value in output.items():
                        if key == target_dict_key:
                            output_coll.append(output[target_dict_key])
                        elif isinstance(value, list) and all((isinstance(elem, dict) for elem in value)):
                            output_coll.extend(self._parse_lists_of_dict_to_list_of_str(value, target_dict_key))
                elif isinstance(output, list):
                    output_coll.extend(self._parse_lists_of_dict_to_list_of_str(output, target_dict_key))
            return output_coll
        elif target_dict_key:
            return output_data[target_dict_key]
        else:
            return output_data

    @staticmethod
    def _parse_feature_extraction_input(input_data):
        if isinstance(input_data, list) and isinstance(input_data[0], dict):
            return [list(data.values())[0] for data in input_data]
        else:
            return input_data

    @staticmethod
    def _parse_feature_extraction_output(output_data):
        """
        Parse the return type from a FeatureExtractionPipeline output. The mixed types for
        input are present depending on how the pyfunc is instantiated. For model serving usage,
        the returned type from MLServer will be a numpy.ndarray type, otherwise, the return
        within a manually executed pyfunc (i.e., for udf usage), the return will be a collection
        of nested lists.

        Examples:

        Input: [[[0.11, 0.98, 0.76]]] or np.array([0.11, 0.98, 0.76])
        Output: np.array([0.11, 0.98, 0.76])

        Input: [[[[0.1, 0.2], [0.3, 0.4]]]] or
            np.array([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        Output: np.array([np.array([0.1, 0.2]), np.array([0.3, 0.4])])
        """
        if isinstance(output_data, np.ndarray):
            return output_data
        else:
            return np.array(output_data[0][0])

    def _parse_tokenizer_output(self, output_data, target_set):
        """
        Parses the tokenizer pipeline output.

        Examples:

        Input: [{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}]
        Output: "PRON,NOUN"

        Input: [[{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}],
                [{"entity": "PRON", "score": 0.95}, {"entity": "NOUN", "score": 0.998}]]
        Output: ["PRON,NOUN", "PRON,NOUN"]
        """
        if isinstance(output_data[0], list):
            return [self._parse_tokenizer_output(coll, target_set) for coll in output_data]
        else:
            target = target_set.intersection(output_data[0].keys()).pop()
            return ','.join([coll[target] for coll in output_data])

    @staticmethod
    def _parse_list_of_multiple_dicts(output_data, target_dict_key):
        """
        Returns the first value of the `target_dict_key` that matches in the first dictionary in a
        list of dictionaries.
        """

        def fetch_target_key_value(data, key):
            if isinstance(data[0], dict):
                return data[0][key]
            return [item[0][key] for item in data]
        if isinstance(output_data[0], list):
            return [fetch_target_key_value(collection, target_dict_key) for collection in output_data]
        else:
            return [output_data[0][target_dict_key]]

    def _parse_list_output_for_multiple_candidate_pipelines(self, output_data):
        if isinstance(output_data, list) and len(output_data) < 1:
            raise MlflowException('The output of the pipeline contains no data.', error_code=BAD_REQUEST)
        if isinstance(output_data[0], list):
            return [self._parse_list_output_for_multiple_candidate_pipelines(x) for x in output_data]
        else:
            return output_data[0]

    def _parse_question_answer_input(self, data):
        """
        Parses the single string input representation for a question answer pipeline into the
        required dict format for a `question-answering` pipeline.
        """
        if isinstance(data, list):
            return [self._parse_question_answer_input(entry) for entry in data]
        elif isinstance(data, dict):
            expected_keys = {'question', 'context'}
            if not expected_keys.intersection(set(data.keys())) == expected_keys:
                raise MlflowException(f'Invalid keys were submitted. Keys must be exclusively {expected_keys}')
            return data
        else:
            raise MlflowException(f'An invalid type has been supplied. Must be either List[Dict[str, str]] or Dict[str, str]. {type(data)} is not supported.', error_code=INVALID_PARAMETER_VALUE)

    def _parse_text2text_input(self, data):
        """
        Parses the mixed input types that can be submitted into a text2text Pipeline.
        Valid examples:

        Input:
            {"context": "abc", "answer": "def"}
        Output:
            "context: abc answer: def"
        Input:
            [{"context": "abc", "answer": "def"}, {"context": "ghi", "answer": "jkl"}]
        Output:
            ["context: abc answer: def", "context: ghi answer: jkl"]
        Input:
            "abc"
        Output:
            "abc"
        Input:
            ["abc", "def"]
        Output:
            ["abc", "def"]
        """
        if isinstance(data, dict) and all((isinstance(value, str) for value in data.values())):
            if all((isinstance(key, str) for key in data)) and 'inputs' not in data:
                return ' '.join((f'{key}: {value}' for key, value in data.items()))
            else:
                return list(data.values())
        elif isinstance(data, list) and all((isinstance(value, dict) for value in data)):
            return [self._parse_text2text_input(entry) for entry in data]
        elif isinstance(data, str) or (isinstance(data, list) and all((isinstance(value, str) for value in data))):
            return data
        else:
            raise MlflowException('An invalid type has been supplied. Please supply a Dict[str, str], str, List[str], or a List[Dict[str, str]] for a Text2Text Pipeline.', error_code=INVALID_PARAMETER_VALUE)

    def _parse_json_encoded_list(self, data, key_to_unpack):
        """
        Parses the complex input types for pipelines such as ZeroShotClassification in which
        the required input type is Dict[str, Union[str, List[str]]] wherein the list
        provided is encoded as JSON. This method unpacks that string to the required
        elements.
        """
        if isinstance(data, list):
            return [self._parse_json_encoded_list(entry, key_to_unpack) for entry in data]
        elif isinstance(data, dict):
            if key_to_unpack not in data:
                raise MlflowException(f'Invalid key in inference payload. The expected inference data key is: {key_to_unpack}', error_code=INVALID_PARAMETER_VALUE)
            if isinstance(data[key_to_unpack], str):
                try:
                    return {k: json.loads(v) if k == key_to_unpack else v for k, v in data.items()}
                except json.JSONDecodeError:
                    return data
            elif isinstance(data[key_to_unpack], list):
                return data

    @staticmethod
    def _parse_json_encoded_dict_payload_to_dict(data, key_to_unpack):
        """
        Parses complex dict input types that have been json encoded. Pipelines like
        TableQuestionAnswering require such input types.
        """
        if isinstance(data, list):
            return [{key: json.loads(value) if key == key_to_unpack and isinstance(value, str) else value for key, value in entry.items()} for entry in data]
        elif isinstance(data, dict):
            output = {}
            for key, value in data.items():
                if key == key_to_unpack:
                    if isinstance(value, np.ndarray):
                        output[key] = ast.literal_eval(value.item())
                    else:
                        output[key] = ast.literal_eval(value)
                elif isinstance(value, np.ndarray):
                    output[key] = value.item()
                else:
                    output[key] = value
            return output
        else:
            return {key: json.loads(value) if key == key_to_unpack and isinstance(value, str) else value for key, value in data.items()}

    @staticmethod
    def _validate_str_or_list_str(data):
        if not isinstance(data, (str, list)):
            raise MlflowException(f'The input data is of an incorrect type. {type(data)} is invalid. Must be either string or List[str]', error_code=INVALID_PARAMETER_VALUE)
        elif isinstance(data, list) and (not all((isinstance(entry, str) for entry in data))):
            raise MlflowException('If supplying a list, all values must be of string type.', error_code=INVALID_PARAMETER_VALUE)

    @staticmethod
    def _convert_cast_lists_from_np_back_to_list(data):
        """
        This handles the casting of dicts within lists from Pandas DF conversion within model
        serving back into the required Dict[str, List[str]] if this type matching occurs.
        Otherwise, it's a noop.
        """
        if not isinstance(data, list):
            return data
        elif not all((isinstance(value, dict) for value in data)):
            return data
        else:
            parsed_data = []
            for entry in data:
                if all((isinstance(value, np.ndarray) for value in entry.values())):
                    parsed_data.append({key: value.tolist() for key, value in entry.items()})
                else:
                    parsed_data.append(entry)
            return parsed_data

    @staticmethod
    def is_base64_image(image):
        """Check whether input image is a base64 encoded"""
        try:
            return base64.b64encode(base64.b64decode(image)).decode('utf-8') == image
        except binascii.Error:
            return False

    def _convert_image_input(self, input_data):
        """
        Conversion utility for decoding the base64 encoded bytes data of a raw image file when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.images/a/sound/file.jpg")
        encoded_image = base64.b64encode(response.content).decode("utf-8")

        inference_data = json.dumps({"inputs": [encoded_image]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_image], name="image_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the Image pipelines for inference.
        """

        def process_input_element(input_element):
            input_value = next(iter(input_element.values()))
            if isinstance(input_value, str) and (not self.is_base64_image(input_value)):
                self._validate_str_input_uri_or_file(input_value)
            return input_value
        if isinstance(input_data, list) and all((isinstance(element, dict) for element in input_data)):
            return [process_input_element(element) for element in input_data]
        elif isinstance(input_data, str) and (not self.is_base64_image(input_data)):
            self._validate_str_input_uri_or_file(input_data)
        return input_data

    def _convert_audio_input(self, data):
        """
        Conversion utility for decoding the base64 encoded bytes data of a raw soundfile when
        parsed through model serving, if applicable. Direct usage of the pyfunc implementation
        outside of model serving will treat this utility as a noop.

        For reference, the expected encoding for input to Model Serving will be:

        import requests
        import base64

        response = requests.get("https://www.my.sound/a/sound/file.wav")
        encoded_audio = base64.b64encode(response.content).decode("ascii")

        inference_data = json.dumps({"inputs": [encoded_audio]})

        or

        inference_df = pd.DataFrame(
        pd.Series([encoded_audio], name="audio_file")
        )
        split_dict = {"dataframe_split": inference_df.to_dict(orient="split")}
        split_json = json.dumps(split_dict)

        or

        records_dict = {"dataframe_records": inference_df.to_dict(orient="records")}
        records_json = json.dumps(records_dict)

        This utility will convert this JSON encoded, base64 encoded text back into bytes for
        input into the AutomaticSpeechRecognitionPipeline for inference.
        """

        def is_base64(s):
            try:
                return base64.b64encode(base64.b64decode(s)) == s
            except binascii.Error:
                return False

        def decode_audio(encoded):
            if isinstance(encoded, str):
                return encoded
            elif isinstance(encoded, bytes):
                if not is_base64(encoded):
                    return encoded
                else:
                    return base64.b64decode(encoded)
            else:
                try:
                    return base64.b64decode(encoded)
                except binascii.Error as e:
                    raise MlflowException("The encoded soundfile that was passed has not been properly base64 encoded. Please ensure that the raw sound bytes have been processed with `base64.b64encode(<audio data bytes>).decode('ascii')`") from e
        if isinstance(data, list) and all((isinstance(element, dict) for element in data)):
            encoded_audio = list(data[0].values())[0]
            if isinstance(encoded_audio, str):
                self._validate_str_input_uri_or_file(encoded_audio)
            return decode_audio(encoded_audio)
        elif isinstance(data, str):
            self._validate_str_input_uri_or_file(data)
        elif isinstance(data, bytes):
            return decode_audio(data)
        return data

    @staticmethod
    def _validate_str_input_uri_or_file(input_str):
        """
        Validation of blob references to either audio or image files,
        if a string is input to the ``predict``
        method, perform validation of the string contents by checking for a valid uri or
        filesystem reference instead of surfacing the cryptic stack trace that is otherwise raised
        for an invalid uri input.
        """

        def is_uri(s):
            try:
                result = urlparse(s)
                return all([result.scheme, result.netloc])
            except ValueError:
                return False
        valid_uri = os.path.isfile(input_str) or is_uri(input_str)
        if not valid_uri:
            if len(input_str) <= 20:
                data_str = f'Received: {input_str}'
            else:
                data_str = f'Received (truncated): {input_str[:20]}...'
            raise MlflowException(f'An invalid string input was provided. String inputs to audio or image files must be either a file location or a uri.audio files must be either a file location or a uri. {data_str}', error_code=BAD_REQUEST)

    def _format_prompt_template(self, input_data):
        """
        Wraps the input data in the specified prompt template. If no template is
        specified, or if the pipeline is an unsupported type, or if the input type
        is not a string or list of strings, then the input data is returned unchanged.
        """
        if not self.prompt_template:
            return input_data
        if self.pipeline.task not in _SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES:
            raise MlflowException(f'_format_prompt_template called on an unexpected pipeline type. Expected one of: {_SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES}. Received: {self.pipeline.task}')
        if isinstance(input_data, str):
            return self.prompt_template.format(prompt=input_data)
        elif isinstance(input_data, list):
            if all((isinstance(data, str) for data in input_data)):
                return [self.prompt_template.format(prompt=data) for data in input_data]
        raise MlflowException.invalid_parameter_value(f'Prompt templating is only supported for data of type str or List[str]. Got {type(input_data)} instead.')