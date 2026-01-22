import requests
from wandb_gql import gql
import wandb
from wandb.apis.attrs import Attrs
@property
def api_keys(self):
    if self._attrs.get('apiKeys') is None:
        return []
    return [k['node']['name'] for k in self._attrs['apiKeys']['edges']]