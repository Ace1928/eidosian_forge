import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('use_threads', [True, False])
def test_output_field_names(use_threads):
    in_table = pa.Table.from_pydict({'x': [1, 2, 3]})

    def table_provider(names, schema):
        return in_table
    substrait_query = '\n    {\n      "version": { "major": 9999 },\n      "relations": [\n        {\n          "root": {\n            "input": {\n              "read": {\n                "base_schema": {\n                  "struct": {\n                    "types": [{"i64": {}}]\n                  },\n                  "names": ["x"]\n                },\n                "namedTable": {\n                  "names": ["t1"]\n                }\n              }\n            },\n            "names": ["out"]\n          }\n        }\n      ]\n    }\n    '
    buf = pa._substrait._parse_json_plan(tobytes(substrait_query))
    reader = pa.substrait.run_query(buf, table_provider=table_provider, use_threads=use_threads)
    res_tb = reader.read_all()
    expected = pa.Table.from_pydict({'out': [1, 2, 3]})
    assert res_tb == expected