import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
class WebDataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 100
    IMAGE_EXTENSIONS: List[str]
    AUDIO_EXTENSIONS: List[str]
    DECODERS: Dict[str, Callable[[Any], Any]]
    NUM_EXAMPLES_FOR_FEATURES_INFERENCE = 5

    @classmethod
    def _get_pipeline_from_tar(cls, tar_path, tar_iterator):
        current_example = {}
        for filename, f in tar_iterator:
            if '.' in filename:
                example_key, field_name = filename.split('.', 1)
                if current_example and current_example['__key__'] != example_key:
                    yield current_example
                    current_example = {}
                current_example['__key__'] = example_key
                current_example['__url__'] = tar_path
                current_example[field_name.lower()] = f.read()
                if field_name in cls.DECODERS:
                    current_example[field_name] = cls.DECODERS[field_name](current_example[field_name])
        if current_example:
            yield current_example

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo()

    def _split_generators(self, dl_manager):
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(f'At least one data file must be specified, but got data_files={self.config.data_files}')
        data_files = dl_manager.download(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            tar_paths = data_files
            if isinstance(tar_paths, str):
                tar_paths = [tar_paths]
            tar_iterators = [dl_manager.iter_archive(tar_path) for tar_path in tar_paths]
            splits = [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'tar_paths': tar_paths, 'tar_iterators': tar_iterators})]
        else:
            splits = []
            for split_name, tar_paths in data_files.items():
                if isinstance(tar_paths, str):
                    tar_paths = [tar_paths]
                tar_iterators = [dl_manager.iter_archive(tar_path) for tar_path in tar_paths]
                splits.append(datasets.SplitGenerator(name=split_name, gen_kwargs={'tar_paths': tar_paths, 'tar_iterators': tar_iterators}))
        if not self.info.features:
            pipeline = self._get_pipeline_from_tar(tar_paths[0], tar_iterators[0])
            first_examples = list(islice(pipeline, self.NUM_EXAMPLES_FOR_FEATURES_INFERENCE))
            if any((example.keys() != first_examples[0].keys() for example in first_examples)):
                raise ValueError("The TAR archives of the dataset should be in WebDataset format, but the files in the archive don't share the same prefix or the same types.")
            pa_tables = [pa.Table.from_pylist([example]) for example in first_examples]
            if datasets.config.PYARROW_VERSION.major < 14:
                inferred_arrow_schema = pa.concat_tables(pa_tables, promote=True).schema
            else:
                inferred_arrow_schema = pa.concat_tables(pa_tables, promote_options='default').schema
            features = datasets.Features.from_arrow_schema(inferred_arrow_schema)
            for field_name in first_examples[0]:
                extension = field_name.rsplit('.', 1)[-1]
                if extension in self.IMAGE_EXTENSIONS:
                    features[field_name] = datasets.Image()
            for field_name in first_examples[0]:
                extension = field_name.rsplit('.', 1)[-1]
                if extension in self.AUDIO_EXTENSIONS:
                    features[field_name] = datasets.Audio()
            self.info.features = features
        return splits

    def _generate_examples(self, tar_paths, tar_iterators):
        image_field_names = [field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Image)]
        audio_field_names = [field_name for field_name, feature in self.info.features.items() if isinstance(feature, datasets.Audio)]
        for tar_idx, (tar_path, tar_iterator) in enumerate(zip(tar_paths, tar_iterators)):
            for example_idx, example in enumerate(self._get_pipeline_from_tar(tar_path, tar_iterator)):
                for field_name in image_field_names + audio_field_names:
                    example[field_name] = {'path': example['__key__'] + '.' + field_name, 'bytes': example[field_name]}
                yield (f'{tar_idx}_{example_idx}', example)