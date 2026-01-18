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
def write_component_resources(name, component):
    write_bundled_files(name, list(component._resources.get('css', {}).values()), BUNDLE_DIR, 'css')
    write_bundled_files(name, list(component._resources.get('js', {}).values()), BUNDLE_DIR, 'js')
    js_modules = []
    for tar_name, js_module in component._resources.get('js_modules', {}).items():
        if tar_name not in component._resources.get('tarball', {}):
            js_modules.append(js_module)
    write_bundled_files(name, js_modules, 'js', ext='mjs')
    for tarball in component._resources.get('tarball', {}).values():
        write_bundled_tarball(tarball)