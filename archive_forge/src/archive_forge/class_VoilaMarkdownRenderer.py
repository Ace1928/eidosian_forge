import mimetypes
from typing import Optional
import traitlets
from traitlets.config import Config
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.exporters.html import HTMLExporter
from nbconvert.exporters.templateexporter import TemplateExporter
from nbconvert.filters.highlight import Highlight2HTML
from .static_file_handler import TemplateStaticFileHandler
from .utils import create_include_assets_functions
class VoilaMarkdownRenderer(IPythonRenderer):
    """Custom markdown renderer that inlines images"""

    def __init__(self, contents_manager, *args, **kwargs):
        self.contents_manager = contents_manager
        super().__init__(*args, **kwargs)

    def image(self, text: str, url: str, title: Optional[str]=None):
        contents_manager = self.contents_manager
        src = url if NB_CONVERT_760 else text
        if contents_manager.file_exists(src):
            content = contents_manager.get(src, format='base64')
            data = content['content'].replace('\n', '')
            mime_type, encoding = mimetypes.guess_type(src)
            src = f'data:{mime_type};base64,{data}'
        if NB_CONVERT_760:
            return super().image(text, src, title)
        else:
            return super().image(src, url, title)