import logging
from pathlib import Path
from typing import List, Union
from transformers import AutoFeatureExtractor, AutoProcessor, AutoTokenizer

    Saves the tokenizer, the processor and the feature extractor when found in `src_dir` in `dest_dir`.

    Args:
        src_dir (`Union[str, Path]`):
            The source directory from which to copy the files.
        dest_dir (`Union[str, Path]`):
            The destination directory to copy the files to.
        src_subfolder (`str`, defaults to `""`):
            In case the preprocessor files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        trust_remote_code (`bool`, defaults to `False`):
            Whether to allow to save preprocessors that is allowed to run arbitrary code. Use this option at your own risk.
    