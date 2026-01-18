from typing import Any, Callable, Dict, Optional, List
import ibis
import ibis.expr.datatypes as dt
import pyarrow as pa
from triad import Schema, extensible_class
from triad.utils.pyarrow import TRIAD_DEFAULT_TIMESTAMP
def to_ibis_schema(schema: Schema) -> ibis.Schema:
    fields = [(f.name, pa_to_ibis_type(f.type)) for f in schema.fields]
    return ibis.schema(fields)