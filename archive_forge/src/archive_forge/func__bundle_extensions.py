from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
from dataclasses import dataclass
from os.path import normpath
from pathlib import Path
from typing import (
from urllib.parse import urljoin
from ..core.has_props import HasProps
from ..core.templates import CSS_RESOURCES, JS_RESOURCES
from ..document.document import Document
from ..resources import Resources
from ..settings import settings
from ..util.compiler import bundle_models
from .util import contains_tex_string
def _bundle_extensions(objs: set[HasProps] | None, resources: Resources) -> list[ExtensionEmbed]:
    names: set[str] = set()
    bundles: list[ExtensionEmbed] = []
    extensions = ['.min.js', '.js'] if resources.minified else ['.js']
    all_objs = objs if objs is not None else HasProps.model_class_reverse_map.values()
    for obj in all_objs:
        if hasattr(obj, '__implementation__'):
            continue
        name = obj.__view_module__.split('.')[0]
        if name == 'bokeh':
            continue
        if name in names:
            continue
        names.add(name)
        module = __import__(name)
        this_file = Path(module.__file__).absolute()
        base_dir = this_file.parent
        dist_dir = base_dir / 'dist'
        ext_path = base_dir / 'bokeh.ext.json'
        if not ext_path.exists():
            continue
        server_prefix = URL(resources.root_url) / 'static' / 'extensions'
        package_path = base_dir / 'package.json'
        pkg: Pkg | None = None
        if package_path.exists():
            with open(package_path) as io:
                try:
                    pkg = json.load(io)
                except json.decoder.JSONDecodeError:
                    pass
        artifact_path: Path
        server_url: URL
        cdn_url: URL | None = None
        if pkg is not None:
            pkg_name: str | None = pkg.get('name', None)
            if pkg_name is None:
                raise ValueError('invalid package.json; missing package name')
            pkg_version = pkg.get('version', 'latest')
            pkg_main = pkg.get('module', pkg.get('main', None))
            if pkg_main is not None:
                pkg_main = Path(normpath(pkg_main))
                cdn_url = _default_cdn_host / f'{pkg_name}@{pkg_version}' / f'{pkg_main}'
            else:
                pkg_main = dist_dir / f'{name}.js'
            artifact_path = base_dir / pkg_main
            artifacts_dir = artifact_path.parent
            artifact_name = artifact_path.name
            server_path = f'{name}/{artifact_name}'
            if not settings.dev:
                sha = hashlib.sha256()
                sha.update(pkg_version.encode())
                vstring = sha.hexdigest()
                server_path = f'{server_path}?v={vstring}'
        else:
            for ext in extensions:
                artifact_path = dist_dir / f'{name}{ext}'
                artifacts_dir = dist_dir
                server_path = f'{name}/{name}{ext}'
                if artifact_path.exists():
                    break
            else:
                raise ValueError(f"can't resolve artifact path for '{name}' extension")
        extension_dirs[name] = Path(artifacts_dir)
        server_url = server_prefix / server_path
        embed = ExtensionEmbed(artifact_path, server_url, cdn_url)
        bundles.append(embed)
    return bundles