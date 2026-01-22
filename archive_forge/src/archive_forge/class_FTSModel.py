import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
class FTSModel(BaseFTSModel):
    """
    VirtualModel class for creating tables that use either the FTS3 or FTS4
    search extensions. Peewee automatically determines which version of the
    FTS extension is supported and will use FTS4 if possible.
    """
    docid = DocIDField()

    class Meta:
        extension_module = 'FTS%s' % FTS_VERSION

    @classmethod
    def _fts_cmd(cls, cmd):
        tbl = cls._meta.table_name
        res = cls._meta.database.execute_sql("INSERT INTO %s(%s) VALUES('%s');" % (tbl, tbl, cmd))
        return res.fetchone()

    @classmethod
    def optimize(cls):
        return cls._fts_cmd('optimize')

    @classmethod
    def rebuild(cls):
        return cls._fts_cmd('rebuild')

    @classmethod
    def integrity_check(cls):
        return cls._fts_cmd('integrity-check')

    @classmethod
    def merge(cls, blocks=200, segments=8):
        return cls._fts_cmd('merge=%s,%s' % (blocks, segments))

    @classmethod
    def automerge(cls, state=True):
        return cls._fts_cmd('automerge=%s' % (state and '1' or '0'))

    @classmethod
    def match(cls, term):
        """
        Generate a `MATCH` expression appropriate for searching this table.
        """
        return match(cls._meta.entity, term)

    @classmethod
    def rank(cls, *weights):
        matchinfo = fn.matchinfo(cls._meta.entity, FTS3_MATCHINFO)
        return fn.fts_rank(matchinfo, *weights)

    @classmethod
    def bm25(cls, *weights):
        match_info = fn.matchinfo(cls._meta.entity, FTS4_MATCHINFO)
        return fn.fts_bm25(match_info, *weights)

    @classmethod
    def bm25f(cls, *weights):
        match_info = fn.matchinfo(cls._meta.entity, FTS4_MATCHINFO)
        return fn.fts_bm25f(match_info, *weights)

    @classmethod
    def lucene(cls, *weights):
        match_info = fn.matchinfo(cls._meta.entity, FTS4_MATCHINFO)
        return fn.fts_lucene(match_info, *weights)

    @classmethod
    def _search(cls, term, weights, with_score, score_alias, score_fn, explicit_ordering):
        if not weights:
            rank = score_fn()
        elif isinstance(weights, dict):
            weight_args = []
            for field in cls._meta.sorted_fields:
                field_weight = weights.get(field, weights.get(field.name, 1.0))
                weight_args.append(field_weight)
            rank = score_fn(*weight_args)
        else:
            rank = score_fn(*weights)
        selection = ()
        order_by = rank
        if with_score:
            selection = (cls, rank.alias(score_alias))
        if with_score and (not explicit_ordering):
            order_by = SQL(score_alias)
        return cls.select(*selection).where(cls.match(term)).order_by(order_by)

    @classmethod
    def search(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search using selected `term`."""
        return cls._search(term, weights, with_score, score_alias, cls.rank, explicit_ordering)

    @classmethod
    def search_bm25(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search for selected `term` using BM25 algorithm."""
        return cls._search(term, weights, with_score, score_alias, cls.bm25, explicit_ordering)

    @classmethod
    def search_bm25f(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search for selected `term` using BM25 algorithm."""
        return cls._search(term, weights, with_score, score_alias, cls.bm25f, explicit_ordering)

    @classmethod
    def search_lucene(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search for selected `term` using BM25 algorithm."""
        return cls._search(term, weights, with_score, score_alias, cls.lucene, explicit_ordering)