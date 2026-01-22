from typing import List, Optional, Union
def _get_args_tags(self) -> List[str]:
    args = []
    if self._no_content:
        args.append('NOCONTENT')
    if self._fields:
        args.append('INFIELDS')
        args.append(len(self._fields))
        args += self._fields
    if self._verbatim:
        args.append('VERBATIM')
    if self._no_stopwords:
        args.append('NOSTOPWORDS')
    if self._filters:
        for flt in self._filters:
            if not isinstance(flt, Filter):
                raise AttributeError('Did not receive a Filter object.')
            args += flt.args
    if self._with_payloads:
        args.append('WITHPAYLOADS')
    if self._scorer:
        args += ['SCORER', self._scorer]
    if self._with_scores:
        args.append('WITHSCORES')
    if self._ids:
        args.append('INKEYS')
        args.append(len(self._ids))
        args += self._ids
    if self._slop >= 0:
        args += ['SLOP', self._slop]
    if self._timeout is not None:
        args += ['TIMEOUT', self._timeout]
    if self._in_order:
        args.append('INORDER')
    if self._return_fields:
        args.append('RETURN')
        args.append(len(self._return_fields))
        args += self._return_fields
    if self._sortby:
        if not isinstance(self._sortby, SortbyField):
            raise AttributeError('Did not receive a SortByField.')
        args.append('SORTBY')
        args += self._sortby.args
    if self._language:
        args += ['LANGUAGE', self._language]
    if self._expander:
        args += ['EXPANDER', self._expander]
    if self._dialect:
        args += ['DIALECT', self._dialect]
    return args