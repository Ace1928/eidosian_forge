import logging
import os
import torch
from torchaudio._internal import download_url_to_file, module_utils as _mod_utils
def _load_phonemizer(file, dl_kwargs):
    if not _mod_utils.is_module_available('dp'):
        raise RuntimeError('DeepPhonemizer is not installed. Please install it.')
    from dp.phonemizer import Phonemizer
    logger = logging.getLogger('dp')
    orig_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        url = f'https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/{file}'
        directory = os.path.join(torch.hub.get_dir(), 'checkpoints')
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, file)
        if not os.path.exists(path):
            dl_kwargs = {} if dl_kwargs is None else dl_kwargs
            download_url_to_file(url, path, **dl_kwargs)
        return Phonemizer.from_checkpoint(path)
    finally:
        logger.setLevel(orig_level)