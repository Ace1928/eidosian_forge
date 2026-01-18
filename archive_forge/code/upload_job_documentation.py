import asyncio
import logging
import os
from typing import TYPE_CHECKING, Optional
import wandb
from wandb.sdk.lib.paths import LogicalPath
A file uploader.

        Arguments:
            push_function: function(save_name, actual_path) which actually uploads
                the file.
            save_name: string logical location of the file relative to the run
                directory.
            path: actual string path of the file to upload on the filesystem.
        