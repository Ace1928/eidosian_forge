import copy
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests
import yaml
from huggingface_hub import model_info
from huggingface_hub.utils import HFValidationError
from . import __version__
from .models.auto.modeling_auto import (
from .training_args import ParallelMode
from .utils import (
@dataclass
class TrainingSummary:
    model_name: str
    language: Optional[Union[str, List[str]]] = None
    license: Optional[str] = None
    tags: Optional[Union[str, List[str]]] = None
    finetuned_from: Optional[str] = None
    tasks: Optional[Union[str, List[str]]] = None
    dataset: Optional[Union[str, List[str]]] = None
    dataset_tags: Optional[Union[str, List[str]]] = None
    dataset_args: Optional[Union[str, List[str]]] = None
    dataset_metadata: Optional[Dict[str, Any]] = None
    eval_results: Optional[Dict[str, float]] = None
    eval_lines: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    source: Optional[str] = 'trainer'

    def __post_init__(self):
        if self.license is None and (not is_offline_mode()) and (self.finetuned_from is not None) and (len(self.finetuned_from) > 0):
            try:
                info = model_info(self.finetuned_from)
                for tag in info.tags:
                    if tag.startswith('license:'):
                        self.license = tag[8:]
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, HFValidationError):
                pass

    def create_model_index(self, metric_mapping):
        model_index = {'name': self.model_name}
        dataset_names = _listify(self.dataset)
        dataset_tags = _listify(self.dataset_tags)
        dataset_args = _listify(self.dataset_args)
        dataset_metadata = _listify(self.dataset_metadata)
        if len(dataset_args) < len(dataset_tags):
            dataset_args = dataset_args + [None] * (len(dataset_tags) - len(dataset_args))
        dataset_mapping = dict(zip(dataset_tags, dataset_names))
        dataset_arg_mapping = dict(zip(dataset_tags, dataset_args))
        dataset_metadata_mapping = dict(zip(dataset_tags, dataset_metadata))
        task_mapping = {task: TASK_TAG_TO_NAME_MAPPING[task] for task in _listify(self.tasks) if task in TASK_TAG_TO_NAME_MAPPING}
        model_index['results'] = []
        if len(task_mapping) == 0 and len(dataset_mapping) == 0:
            return [model_index]
        if len(task_mapping) == 0:
            task_mapping = {None: None}
        if len(dataset_mapping) == 0:
            dataset_mapping = {None: None}
        all_possibilities = [(task_tag, ds_tag) for task_tag in task_mapping for ds_tag in dataset_mapping]
        for task_tag, ds_tag in all_possibilities:
            result = {}
            if task_tag is not None:
                result['task'] = {'name': task_mapping[task_tag], 'type': task_tag}
            if ds_tag is not None:
                metadata = dataset_metadata_mapping.get(ds_tag, {})
                result['dataset'] = {'name': dataset_mapping[ds_tag], 'type': ds_tag, **metadata}
                if dataset_arg_mapping[ds_tag] is not None:
                    result['dataset']['args'] = dataset_arg_mapping[ds_tag]
            if len(metric_mapping) > 0:
                result['metrics'] = []
                for metric_tag, metric_name in metric_mapping.items():
                    result['metrics'].append({'name': metric_name, 'type': metric_tag, 'value': self.eval_results[metric_name]})
            if 'task' in result and 'dataset' in result and ('metrics' in result):
                model_index['results'].append(result)
            else:
                logger.info(f'Dropping the following result as it does not have all the necessary fields:\n{result}')
        return [model_index]

    def create_metadata(self):
        metric_mapping = infer_metric_tags_from_eval_results(self.eval_results)
        metadata = {}
        metadata = _insert_values_as_list(metadata, 'language', self.language)
        metadata = _insert_value(metadata, 'license', self.license)
        if self.finetuned_from is not None and isinstance(self.finetuned_from, str) and (len(self.finetuned_from) > 0):
            metadata = _insert_value(metadata, 'base_model', self.finetuned_from)
        metadata = _insert_values_as_list(metadata, 'tags', self.tags)
        metadata = _insert_values_as_list(metadata, 'datasets', self.dataset_tags)
        metadata = _insert_values_as_list(metadata, 'metrics', list(metric_mapping.keys()))
        metadata['model-index'] = self.create_model_index(metric_mapping)
        return metadata

    def to_model_card(self):
        model_card = ''
        metadata = yaml.dump(self.create_metadata(), sort_keys=False)
        if len(metadata) > 0:
            model_card = f'---\n{metadata}---\n'
        if self.source == 'trainer':
            model_card += AUTOGENERATED_TRAINER_COMMENT
        else:
            model_card += AUTOGENERATED_KERAS_COMMENT
        model_card += f'\n# {self.model_name}\n\n'
        if self.finetuned_from is None:
            model_card += 'This model was trained from scratch on '
        else:
            model_card += f'This model is a fine-tuned version of [{self.finetuned_from}](https://huggingface.co/{self.finetuned_from}) on '
        if self.dataset is None:
            model_card += 'an unknown dataset.'
        elif isinstance(self.dataset, str):
            model_card += f'the {self.dataset} dataset.'
        elif isinstance(self.dataset, (tuple, list)) and len(self.dataset) == 1:
            model_card += f'the {self.dataset[0]} dataset.'
        else:
            model_card += ', '.join([f'the {ds}' for ds in self.dataset[:-1]]) + f' and the {self.dataset[-1]} datasets.'
        if self.eval_results is not None:
            model_card += '\nIt achieves the following results on the evaluation set:\n'
            model_card += '\n'.join([f'- {name}: {_maybe_round(value)}' for name, value in self.eval_results.items()])
        model_card += '\n'
        model_card += '\n## Model description\n\nMore information needed\n'
        model_card += '\n## Intended uses & limitations\n\nMore information needed\n'
        model_card += '\n## Training and evaluation data\n\nMore information needed\n'
        model_card += '\n## Training procedure\n'
        model_card += '\n### Training hyperparameters\n'
        if self.hyperparameters is not None:
            model_card += '\nThe following hyperparameters were used during training:\n'
            model_card += '\n'.join([f'- {name}: {value}' for name, value in self.hyperparameters.items()])
            model_card += '\n'
        else:
            model_card += '\nMore information needed\n'
        if self.eval_lines is not None:
            model_card += '\n### Training results\n\n'
            model_card += make_markdown_table(self.eval_lines)
            model_card += '\n'
        model_card += '\n### Framework versions\n\n'
        model_card += f'- Transformers {__version__}\n'
        if self.source == 'trainer' and is_torch_available():
            import torch
            model_card += f'- Pytorch {torch.__version__}\n'
        elif self.source == 'keras' and is_tf_available():
            import tensorflow as tf
            model_card += f'- TensorFlow {tf.__version__}\n'
        if is_datasets_available():
            import datasets
            model_card += f'- Datasets {datasets.__version__}\n'
        if is_tokenizers_available():
            import tokenizers
            model_card += f'- Tokenizers {tokenizers.__version__}\n'
        return model_card

    @classmethod
    def from_trainer(cls, trainer, language=None, license=None, tags=None, model_name=None, finetuned_from=None, tasks=None, dataset_tags=None, dataset_metadata=None, dataset=None, dataset_args=None):
        one_dataset = trainer.eval_dataset if trainer.eval_dataset is not None else trainer.train_dataset
        if is_hf_dataset(one_dataset) and (dataset_tags is None or dataset_args is None or dataset_metadata is None):
            default_tag = one_dataset.builder_name
            if default_tag not in ['csv', 'json', 'pandas', 'parquet', 'text']:
                if dataset_metadata is None:
                    dataset_metadata = [{'config': one_dataset.config_name, 'split': str(one_dataset.split)}]
                if dataset_tags is None:
                    dataset_tags = [default_tag]
                if dataset_args is None:
                    dataset_args = [one_dataset.config_name]
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags
        if finetuned_from is None and hasattr(trainer.model.config, '_name_or_path') and (not os.path.isdir(trainer.model.config._name_or_path)):
            finetuned_from = trainer.model.config._name_or_path
        if tasks is None:
            model_class_name = trainer.model.__class__.__name__
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task
        if model_name is None:
            model_name = Path(trainer.args.output_dir).name
        if len(model_name) == 0:
            model_name = finetuned_from
        if tags is None:
            tags = ['generated_from_trainer']
        elif isinstance(tags, str) and tags != 'generated_from_trainer':
            tags = [tags, 'generated_from_trainer']
        elif 'generated_from_trainer' not in tags:
            tags.append('generated_from_trainer')
        _, eval_lines, eval_results = parse_log_history(trainer.state.log_history)
        hyperparameters = extract_hyperparameters_from_trainer(trainer)
        return cls(language=language, license=license, tags=tags, model_name=model_name, finetuned_from=finetuned_from, tasks=tasks, dataset=dataset, dataset_tags=dataset_tags, dataset_args=dataset_args, dataset_metadata=dataset_metadata, eval_results=eval_results, eval_lines=eval_lines, hyperparameters=hyperparameters)

    @classmethod
    def from_keras(cls, model, model_name, keras_history=None, language=None, license=None, tags=None, finetuned_from=None, tasks=None, dataset_tags=None, dataset=None, dataset_args=None):
        if dataset is not None:
            if is_hf_dataset(dataset) and (dataset_tags is None or dataset_args is None):
                default_tag = dataset.builder_name
                if default_tag not in ['csv', 'json', 'pandas', 'parquet', 'text']:
                    if dataset_tags is None:
                        dataset_tags = [default_tag]
                    if dataset_args is None:
                        dataset_args = [dataset.config_name]
        if dataset is None and dataset_tags is not None:
            dataset = dataset_tags
        if finetuned_from is None and hasattr(model.config, '_name_or_path') and (not os.path.isdir(model.config._name_or_path)):
            finetuned_from = model.config._name_or_path
        if tasks is None:
            model_class_name = model.__class__.__name__
            for task, mapping in TASK_MAPPING.items():
                if model_class_name in _get_mapping_values(mapping):
                    tasks = task
        if tags is None:
            tags = ['generated_from_keras_callback']
        elif isinstance(tags, str) and tags != 'generated_from_keras_callback':
            tags = [tags, 'generated_from_keras_callback']
        elif 'generated_from_keras_callback' not in tags:
            tags.append('generated_from_keras_callback')
        if keras_history is not None:
            _, eval_lines, eval_results = parse_keras_history(keras_history)
        else:
            eval_lines = []
            eval_results = {}
        hyperparameters = extract_hyperparameters_from_keras(model)
        return cls(language=language, license=license, tags=tags, model_name=model_name, finetuned_from=finetuned_from, tasks=tasks, dataset_tags=dataset_tags, dataset=dataset, dataset_args=dataset_args, eval_results=eval_results, eval_lines=eval_lines, hyperparameters=hyperparameters, source='keras')