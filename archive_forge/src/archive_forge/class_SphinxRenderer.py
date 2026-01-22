import os
from functools import partial
from os import path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from jinja2 import TemplateNotFound
from jinja2.environment import Environment
from jinja2.loaders import BaseLoader
from jinja2.sandbox import SandboxedEnvironment
from sphinx import package_dir
from sphinx.jinja2glue import SphinxFileSystemLoader
from sphinx.locale import get_translator
from sphinx.util import rst, texescape
class SphinxRenderer(FileRenderer):

    def __init__(self, template_path: Union[None, str, List[str]]=None) -> None:
        if template_path is None:
            template_path = os.path.join(package_dir, 'templates')
        super().__init__(template_path)

    @classmethod
    def render_from_file(cls, filename: str, context: Dict[str, Any]) -> str:
        return FileRenderer.render_from_file(filename, context)