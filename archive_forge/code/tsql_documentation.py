from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
Applies to SQL Server and Azure SQL Database
            BEGIN { TRAN | TRANSACTION }
            [ { transaction_name | @tran_name_variable }
            [ WITH MARK [ 'description' ] ]
            ]
            