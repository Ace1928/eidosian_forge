import configparser
import logging
import os
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import wandb
Get the most recent ancestor of HEAD that occurs on an upstream branch.

        First looks at the current branch's tracking branch, if applicable. If
        that doesn't work, looks at every other branch to find the most recent
        ancestor of HEAD that occurs on a tracking branch.

        Returns:
            git.Commit object or None
        