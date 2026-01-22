from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_show_mysql(self, this: str, target: bool | str=False, full: t.Optional[bool]=None, global_: t.Optional[bool]=None) -> exp.Show:
    if target:
        if isinstance(target, str):
            self._match_text_seq(target)
        target_id = self._parse_id_var()
    else:
        target_id = None
    log = self._parse_string() if self._match_text_seq('IN') else None
    if this in ('BINLOG EVENTS', 'RELAYLOG EVENTS'):
        position = self._parse_number() if self._match_text_seq('FROM') else None
        db = None
    else:
        position = None
        db = None
        if self._match(TokenType.FROM):
            db = self._parse_id_var()
        elif self._match(TokenType.DOT):
            db = target_id
            target_id = self._parse_id_var()
    channel = self._parse_id_var() if self._match_text_seq('FOR', 'CHANNEL') else None
    like = self._parse_string() if self._match_text_seq('LIKE') else None
    where = self._parse_where()
    if this == 'PROFILE':
        types = self._parse_csv(lambda: self._parse_var_from_options(self.PROFILE_TYPES))
        query = self._parse_number() if self._match_text_seq('FOR', 'QUERY') else None
        offset = self._parse_number() if self._match_text_seq('OFFSET') else None
        limit = self._parse_number() if self._match_text_seq('LIMIT') else None
    else:
        types, query = (None, None)
        offset, limit = self._parse_oldstyle_limit()
    mutex = True if self._match_text_seq('MUTEX') else None
    mutex = False if self._match_text_seq('STATUS') else mutex
    return self.expression(exp.Show, this=this, target=target_id, full=full, log=log, position=position, db=db, channel=channel, like=like, where=where, types=types, query=query, offset=offset, limit=limit, mutex=mutex, **{'global': global_})