import json
import logging
import os
import re
import shutil
import sys
from base64 import b64encode
from typing import Dict
import requests
from requests.compat import urljoin
import wandb
import wandb.util
from wandb.sdk.lib import filesystem
def _save_ipynb(self) -> bool:
    relpath = self.settings._jupyter_path
    logger.info('looking for notebook: %s', relpath)
    if relpath:
        if os.path.exists(relpath):
            shutil.copy(relpath, os.path.join(self.settings._tmp_code_dir, os.path.basename(relpath)))
            return True
    colab_ipynb = attempt_colab_load_ipynb()
    if colab_ipynb:
        try:
            jupyter_metadata = notebook_metadata_from_jupyter_servers_and_kernel_id()
            nb_name = jupyter_metadata['name']
        except Exception:
            nb_name = 'colab.ipynb'
        if not nb_name.endswith('.ipynb'):
            nb_name += '.ipynb'
        with open(os.path.join(self.settings._tmp_code_dir, nb_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(colab_ipynb))
        return True
    kaggle_ipynb = attempt_kaggle_load_ipynb()
    if kaggle_ipynb and len(kaggle_ipynb['cells']) > 0:
        with open(os.path.join(self.settings._tmp_code_dir, kaggle_ipynb['metadata']['name']), 'w', encoding='utf-8') as f:
            f.write(json.dumps(kaggle_ipynb))
        return True
    return False