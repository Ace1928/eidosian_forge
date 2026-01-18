import getpass
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import click
import requests
from wandb_gql import gql
import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.lib import runid
from ...apis.internal import Api
def verify_digest(downloaded: 'Artifact', computed: 'Artifact', fails_list: List[str]) -> None:
    if downloaded.digest != computed.digest:
        fails_list.append('Artifact digest does not appear as expected. Contact W&B for support.')