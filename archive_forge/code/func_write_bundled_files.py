import fnmatch
import glob
import inspect
import io
import os
import pathlib
import shutil
import tarfile
import zipfile
import param
import requests
from bokeh.model import Model
from .config import config, panel_extension
from .io.resources import RESOURCE_URLS
from .reactive import ReactiveHTML
from .template.base import BasicTemplate
from .theme import Design
def write_bundled_files(name, files, explicit_dir=None, ext=None):
    model_name = name.split('.')[-1].lower()
    for bundle_file in files:
        if not bundle_file.startswith('http'):
            dest_path = BUNDLE_DIR / name.lower() / bundle_file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE_DIR / 'theme' / bundle_file, dest_path)
            continue
        bundle_file = bundle_file.split('?')[0]
        try:
            response = requests.get(bundle_file)
        except Exception:
            try:
                response = requests.get(bundle_file, verify=False)
            except Exception as e:
                raise ConnectionError(f'Failed to fetch {name} dependency: {bundle_file}. Errored with {e}.') from e
        try:
            map_file = f'{bundle_file}.map'
            map_response = requests.get(map_file)
        except Exception:
            try:
                map_response = requests.get(map_file, verify=False)
            except Exception:
                map_response = None
        if bundle_file.startswith(config.npm_cdn):
            bundle_path = os.path.join(*bundle_file.replace(config.npm_cdn, '').split('/'))
        else:
            bundle_path = os.path.join(*os.path.join(*bundle_file.split('//')[1:]).split('/')[1:])
        obj_dir = explicit_dir or model_name
        filename = BUNDLE_DIR.joinpath(obj_dir, bundle_path)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename = str(filename)
        if ext and (not str(filename).endswith(ext)):
            filename += f'.{ext}'
        if filename.endswith(('.ttf', '.wasm')):
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            content = response.content.decode('utf-8')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        if map_response:
            with open(f'{filename}.map', 'w', encoding='utf-8') as f:
                f.write(map_response.content.decode('utf-8'))