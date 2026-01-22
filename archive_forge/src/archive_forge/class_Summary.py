import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
class Summary(SummarySubDict):
    """Store summary metrics (eg. accuracy) during and after a run.

    You can manipulate this as if it's a Python dictionary but the keys
    get mangled. .strip() is called on them, so spaces at the beginning
    and end are removed.
    """

    def __init__(self, run, summary=None):
        super().__init__()
        self._run = run
        self._h5_path = os.path.join(self._run.dir, DEEP_SUMMARY_FNAME)
        self._h5 = None
        self._json_dict = {}
        if summary is not None:
            self._json_dict = summary

    def _json_get(self, path):
        pass

    def _root_get(self, path, child_dict):
        json_dict = self._json_dict
        for key in path[:-1]:
            json_dict = json_dict[key]
        key = path[-1]
        if key in json_dict:
            child_dict[key] = self._decode(path, json_dict[key])

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

    def _root_set(self, path, new_keys_values):
        json_dict = self._json_dict
        for key in path:
            json_dict = json_dict[key]
        for new_key, new_value in new_keys_values:
            json_dict[new_key] = self._encode(new_value, path + (new_key,))

    def write_h5(self, path, val):
        self.open_h5()
        if not self._h5:
            wandb.termerror('Storing tensors in summary requires h5py')
        else:
            try:
                del self._h5['summary/' + '.'.join(path)]
            except KeyError:
                pass
            self._h5['summary/' + '.'.join(path)] = val
            self._h5.flush()

    def read_h5(self, path, val=None):
        self.open_h5()
        if not self._h5:
            wandb.termerror('Reading tensors from summary requires h5py')
        else:
            return self._h5.get('summary/' + '.'.join(path), val)

    def open_h5(self):
        if not self._h5 and h5py:
            self._h5 = h5py.File(self._h5_path, 'a', libver='latest')

    def _decode(self, path, json_value):
        """Decode a `dict` encoded by `Summary._encode()`, loading h5 objects.

        h5 objects may be very large, so we won't have loaded them automatically.
        """
        if isinstance(json_value, dict):
            if json_value.get('_type') in H5_TYPES:
                return self.read_h5(path, json_value)
            elif json_value.get('_type') == 'data-frame':
                wandb.termerror('This data frame was saved via the wandb data API. Contact support@wandb.com for help.')
                return None
            else:
                return SummarySubDict(self, path)
        else:
            return json_value

    def _encode(self, value, path_from_root):
        """Normalize, compress, and encode sub-objects for backend storage.

        value: Object to encode.
        path_from_root: `tuple` of key strings from the top-level summary to the
            current `value`.

        Returns:
            A new tree of dict's with large objects replaced with dictionaries
            with "_type" entries that say which type the original data was.
        """
        if isinstance(value, dict):
            json_value = {}
            for key, value in value.items():
                json_value[key] = self._encode(value, path_from_root + (key,))
            return json_value
        else:
            path = '.'.join(path_from_root)
            friendly_value, converted = util.json_friendly(val_to_json(self._run, path, value, namespace='summary'))
            json_value, compressed = util.maybe_compress_summary(friendly_value, util.get_h5_typename(value))
            if compressed:
                self.write_h5(path_from_root, friendly_value)
            return json_value