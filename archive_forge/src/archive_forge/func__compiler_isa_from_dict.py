from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def _compiler_isa_from_dict(data: Dict[str, Dict[str, Any]]) -> CompilerISA:
    compiler_isa_data = {'1Q': {k: {'id': int(k), **v} for k, v in data.get('1Q', {}).items()}, '2Q': {k: {'ids': _edge_ids_from_id(k), **v} for k, v in data.get('2Q', {}).items()}}
    return CompilerISA.parse_obj(compiler_isa_data)