from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import TYPE_CHECKING, Union, cast
from typing_extensions import TypeAlias
from streamlit.elements.form import current_form_id
from streamlit.elements.utils import (
from streamlit.elements.widgets.file_uploader import _get_upload_files
from streamlit.proto.CameraInput_pb2 import CameraInput as CameraInputProto
from streamlit.proto.Common_pb2 import FileUploaderState as FileUploaderStateProto
from streamlit.proto.Common_pb2 import UploadedFileInfo as UploadedFileInfoProto
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import ScriptRunContext, get_script_run_ctx
from streamlit.runtime.state import (
from streamlit.runtime.state.common import compute_widget_id
from streamlit.runtime.uploaded_file_manager import DeletedFile, UploadedFile
from streamlit.type_util import Key, LabelVisibility, maybe_raise_label_warnings, to_key
def _camera_input(self, label: str, key: Key | None=None, help: str | None=None, on_change: WidgetCallback | None=None, args: WidgetArgs | None=None, kwargs: WidgetKwargs | None=None, *, disabled: bool=False, label_visibility: LabelVisibility='visible', ctx: ScriptRunContext | None=None) -> UploadedFile | None:
    key = to_key(key)
    check_callback_rules(self.dg, on_change)
    check_session_state_rules(default_value=None, key=key, writes_allowed=False)
    maybe_raise_label_warnings(label, label_visibility)
    id = compute_widget_id('camera_input', user_key=key, label=label, key=key, help=help, form_id=current_form_id(self.dg), page=ctx.page_script_hash if ctx else None)
    camera_input_proto = CameraInputProto()
    camera_input_proto.id = id
    camera_input_proto.label = label
    camera_input_proto.form_id = current_form_id(self.dg)
    camera_input_proto.disabled = disabled
    camera_input_proto.label_visibility.value = get_label_visibility_proto_value(label_visibility)
    if help is not None:
        camera_input_proto.help = dedent(help)
    serde = CameraInputSerde()
    camera_input_state = register_widget('camera_input', camera_input_proto, user_key=key, on_change_handler=on_change, args=args, kwargs=kwargs, deserializer=serde.deserialize, serializer=serde.serialize, ctx=ctx)
    self.dg._enqueue('camera_input', camera_input_proto)
    if isinstance(camera_input_state.value, DeletedFile):
        return None
    return camera_input_state.value