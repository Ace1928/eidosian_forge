import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def _root_del(self, path):
    json_dict = self._json_dict
    for key in path[:-1]:
        json_dict = json_dict[key]
    val = json_dict[path[-1]]
    del json_dict[path[-1]]
    if isinstance(val, dict) and val.get('_type') in H5_TYPES:
        if not h5py:
            wandb.termerror('Deleting tensors in summary requires h5py')
        else:
            self.open_h5()
            h5_key = 'summary/' + '.'.join(path)
            del self._h5[h5_key]
            self._h5.flush()