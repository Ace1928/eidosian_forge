from typing import ClassVar, List
import param
from ..io.resources import CDN_DIST
from ..layout import Column
from ..reactive import ReactiveHTML
from ..widgets.base import CompositeWidget
from ..widgets.icon import ToggleIcon
class ChatCopyIcon(ReactiveHTML):
    fill = param.String(default='none', doc='The fill color of the icon.')
    value = param.String(default=None, doc='The text to copy to the clipboard.')
    css_classes = param.List(default=['copy-icon'], doc='The CSS classes of the widget.')
    _template = '\n        <div\n            type="button"\n            id="copy-button"\n            onclick="${script(\'copy_to_clipboard\')}"\n            style="cursor: pointer; width: ${model.width}px; height: ${model.height}px;"\n            title="Copy message to clipboard"\n        >\n            <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-copy" id="copy-icon"\n                width="${model.width}" height="${model.height}" viewBox="0 0 24 24"\n                stroke-width="2" stroke="currentColor" fill=${fill} stroke-linecap="round" stroke-linejoin="round"\n            >\n                <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>\n                <path d="M8 8m0 2a2 2 0 0 1 2 -2h8a2 2 0 0 1 2 2v8a2 2 0 0 1 -2 2h-8a2 2 0 0 1 -2 -2z"></path>\n                <path d="M16 8v-2a2 2 0 0 0 -2 -2h-8a2 2 0 0 0 -2 2v8a2 2 0 0 0 2 2h2"></path>\n            </svg>\n        </div>\n    '
    _scripts = {'copy_to_clipboard': '\n        navigator.clipboard.writeText(`${data.value}`);\n        data.fill = "currentColor";\n        setTimeout(() => data.fill = "none", 50);\n    '}
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/chat_copy_icon.css']