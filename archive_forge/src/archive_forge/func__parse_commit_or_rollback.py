from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_commit_or_rollback(self) -> exp.Commit | exp.Rollback:
    """Applies to SQL Server and Azure SQL Database
            COMMIT [ { TRAN | TRANSACTION }
                [ transaction_name | @tran_name_variable ] ]
                [ WITH ( DELAYED_DURABILITY = { OFF | ON } ) ]

            ROLLBACK { TRAN | TRANSACTION }
                [ transaction_name | @tran_name_variable
                | savepoint_name | @savepoint_variable ]
            """
    rollback = self._prev.token_type == TokenType.ROLLBACK
    self._match_texts(('TRAN', 'TRANSACTION'))
    this = self._parse_id_var()
    if rollback:
        return self.expression(exp.Rollback, this=this)
    durability = None
    if self._match_pair(TokenType.WITH, TokenType.L_PAREN):
        self._match_text_seq('DELAYED_DURABILITY')
        self._match(TokenType.EQ)
        if self._match_text_seq('OFF'):
            durability = False
        else:
            self._match(TokenType.ON)
            durability = True
        self._match_r_paren()
    return self.expression(exp.Commit, this=this, durability=durability)