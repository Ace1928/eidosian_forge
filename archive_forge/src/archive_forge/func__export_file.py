from __future__ import annotations
import glob
import io
import os
import shutil
import zipfile
from typing import Any, List, Mapping, Set, Tuple, Union
import torch
import torch.jit._trace
import torch.serialization
from torch.onnx import _constants, _exporter_states, errors
from torch.onnx._internal import _beartype, jit_utils, registration
@_beartype.beartype
def _export_file(model_bytes: bytes, f: Union[io.BytesIO, str], export_type: str, export_map: Mapping[str, bytes]) -> None:
    """export/write model bytes into directory/protobuf/zip"""
    if export_type == _exporter_states.ExportTypes.PROTOBUF_FILE:
        assert len(export_map) == 0
        with torch.serialization._open_file_like(f, 'wb') as opened_file:
            opened_file.write(model_bytes)
    elif export_type in {_exporter_states.ExportTypes.ZIP_ARCHIVE, _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE}:
        compression = zipfile.ZIP_DEFLATED if export_type == _exporter_states.ExportTypes.COMPRESSED_ZIP_ARCHIVE else zipfile.ZIP_STORED
        with zipfile.ZipFile(f, 'w', compression=compression) as z:
            z.writestr(_constants.ONNX_ARCHIVE_MODEL_PROTO_NAME, model_bytes)
            for k, v in export_map.items():
                z.writestr(k, v)
    elif export_type == _exporter_states.ExportTypes.DIRECTORY:
        if isinstance(f, io.BytesIO) or not os.path.isdir(f):
            raise ValueError(f'f should be directory when export_type is set to DIRECTORY, instead get type(f): {type(f)}')
        if not os.path.exists(f):
            os.makedirs(f)
        model_proto_file = os.path.join(f, _constants.ONNX_ARCHIVE_MODEL_PROTO_NAME)
        with torch.serialization._open_file_like(model_proto_file, 'wb') as opened_file:
            opened_file.write(model_bytes)
        for k, v in export_map.items():
            weight_proto_file = os.path.join(f, k)
            with torch.serialization._open_file_like(weight_proto_file, 'wb') as opened_file:
                opened_file.write(v)
    else:
        raise ValueError('Unknown export type')