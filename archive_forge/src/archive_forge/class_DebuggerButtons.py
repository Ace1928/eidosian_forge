from __future__ import annotations
import logging
from typing import (
import param
from ..io.resources import CDN_DIST
from ..io.state import state
from ..layout import Card, HSpacer, Row
from ..reactive import ReactiveHTML
from .terminal import Terminal
class DebuggerButtons(ReactiveHTML):
    terminal_output = param.String()
    debug_name = param.String()
    clears = param.Integer(default=0)
    _template: ClassVar[str] = '\n    <div style="display: flex;">\n      <button class="special_btn clear_btn" id="clear_btn" onclick="${script(\'click_clear\')}" style="width: ${model.width}px;">\n        <span class="shown">☐</span>\n        <span class="tooltiptext">Acknowledge logs and clear</span>\n      </button>\n      <button class="special_btn" id="save_btn" onclick="${script(\'click\')}" style="width: ${model.width}px;">💾\n        <span class="tooltiptext">Save logs</span>\n      </button>\n    </div>\n    '
    js_cb: ClassVar[str] = '\n        var filename = data.debug_name+\'.txt\'\n        console.log(\'saving debugger terminal output to \'+filename)\n        var blob = new Blob([data.terminal_output],\n            { type: "text/plain;charset=utf-8" });\n        if (navigator.msSaveBlob) {\n            navigator.msSaveBlob(blob, filename);\n        } else {\n            var link = document.createElement(\'a\');\n            var url = URL.createObjectURL(blob);\n            link.href = url;\n            link.download = filename;\n            document.body.appendChild(link);\n            link.click();\n            setTimeout(function() {\n                document.body.removeChild(link);\n                window.URL.revokeObjectURL(url);\n            }, 0);\n        }\n        '
    _scripts: ClassVar[Dict[str, str | List[str]]] = {'click': js_cb, 'click_clear': 'data.clears += 1'}
    _dom_events: ClassVar[Dict[str, List[str]]] = {'clear_btn': ['click']}