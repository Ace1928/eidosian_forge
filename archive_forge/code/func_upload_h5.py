import json
import os
import time
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis.internal import Api
from wandb.sdk import lib as wandb_lib
from wandb.sdk.data_types.utils import val_to_json
def upload_h5(file, run_id, entity=None, project=None):
    api = Api()
    wandb.termlog('Uploading summary data...')
    with open(file, 'rb') as f:
        api.push({os.path.basename(file): f}, run=run_id, project=project, entity=entity)