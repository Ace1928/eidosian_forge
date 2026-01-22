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
class IFrame:

    def __init__(self, path=None, opts=None):
        self.path = path
        self.api = wandb.Api()
        self.opts = opts or {}
        self.displayed = False
        self.height = self.opts.get('height', 420)

    def maybe_display(self) -> bool:
        if not self.displayed and (self.path or wandb.run):
            display(self)
        return self.displayed

    def _repr_html_(self):
        try:
            self.displayed = True
            if self.opts.get('workspace', False):
                if self.path is None and wandb.run:
                    self.path = wandb.run.path
            if isinstance(self.path, str):
                object = self.api.from_path(self.path)
            else:
                object = wandb.run
            if object is None:
                if wandb.Api().api_key is None:
                    return 'You must be logged in to render wandb in jupyter, run `wandb.login()`'
                else:
                    object = self.api.project('/'.join([wandb.Api().default_entity, wandb.util.auto_project_name(None)]))
            return object.to_html(self.height, hidden=False)
        except wandb.Error as e:
            return f"Can't display wandb interface<br/>{e}"