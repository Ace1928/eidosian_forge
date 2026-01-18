from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def tsordstodate_sql(self, expression: exp.TsOrDsToDate) -> str:
    this = expression.this
    time_format = self.format_time(expression)
    if time_format and time_format not in (self.dialect.TIME_FORMAT, self.dialect.DATE_FORMAT):
        return self.sql(exp.cast(exp.StrToTime(this=this, format=expression.args['format']), exp.DataType.Type.DATE))
    if isinstance(this, exp.TsOrDsToDate) or this.is_type(exp.DataType.Type.DATE):
        return self.sql(this)
    return self.sql(exp.cast(this, exp.DataType.Type.DATE))