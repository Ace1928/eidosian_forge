import pathlib
import pytest
from ..filename_parser import (
def test_parse_filename():
    types_exts = (('t1', 'ext1'), ('t2', 'ext2'))
    exp_in_outs = ((('/path/fname.funny', ()), ('/path/fname', '.funny', None, None)), (('/path/fnameext2', ()), ('/path/fname', 'ext2', None, 't2')), (('/path/fnameext2', ('.gz',)), ('/path/fname', 'ext2', None, 't2')), (('/path/fnameext2.gz', ('.gz',)), ('/path/fname', 'ext2', '.gz', 't2')))
    for inps, exps in exp_in_outs:
        pth, sufs = inps
        res = parse_filename(pth, types_exts, sufs)
        assert res == exps
        upth = pth.upper()
        uexps = (exps[0].upper(), exps[1].upper(), exps[2].upper() if exps[2] else None, exps[3])
        res = parse_filename(upth, types_exts, sufs)
        assert res == uexps
        res = parse_filename('/path/fnameext2.GZ', types_exts, ('.gz',), False)
        assert res == ('/path/fname', 'ext2', '.GZ', 't2')
        res = parse_filename('/path/fnameext2.GZ', types_exts, ('.gz',), True)
        assert res == ('/path/fnameext2', '.GZ', None, None)
        res = parse_filename('/path/fnameEXT2.gz', types_exts, ('.gz',), False)
        assert res == ('/path/fname', 'EXT2', '.gz', 't2')
        res = parse_filename('/path/fnameEXT2.gz', types_exts, ('.gz',), True)
        assert res == ('/path/fnameEXT2', '', '.gz', None)