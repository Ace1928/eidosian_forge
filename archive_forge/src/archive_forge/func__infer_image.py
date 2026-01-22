import collections
import importlib
import os
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
from os.path import isfile, join
from urllib.parse import parse_qs
import flask
from . import _validate
from ._utils import AttributeDict
from ._get_paths import get_relative_path
from ._callback_context import context_value
from ._get_app import get_app
def _infer_image(module):
    """
    Return:
    - A page specific image: `assets/<module>.<extension>` is used, e.g. `assets/weekly_analytics.png`
    - A generic app image at `assets/app.<extension>`
    - A logo at `assets/logo.<extension>`
    """
    assets_folder = CONFIG.assets_folder
    valid_extensions = ['apng', 'avif', 'gif', 'jpeg', 'jpg', 'png', 'svg', 'webp']
    page_id = module.split('.')[-1]
    files_in_assets = []
    if os.path.exists(assets_folder):
        files_in_assets = [f for f in os.listdir(assets_folder) if isfile(join(assets_folder, f))]
    app_file = None
    logo_file = None
    for fn in files_in_assets:
        fn_without_extension, _, extension = fn.partition('.')
        if extension.lower() in valid_extensions:
            if fn_without_extension == page_id or fn_without_extension == page_id.replace('_', '-'):
                return fn
            if fn_without_extension == 'app':
                app_file = fn
            if fn_without_extension == 'logo':
                logo_file = fn
    if app_file:
        return app_file
    return logo_file