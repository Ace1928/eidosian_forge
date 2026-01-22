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
class FTS5Model(BaseFTSModel):
    """
    Requires SQLite >= 3.9.0.

    Table options:

    content: table name of external content, or empty string for "contentless"
    content_rowid: column name of external content primary key
    prefix: integer(s). Ex: '2' or '2 3 4'
    tokenize: porter, unicode61, ascii. Ex: 'porter unicode61'

    The unicode tokenizer supports the following parameters:

    * remove_diacritics (1 or 0, default is 1)
    * tokenchars (string of characters, e.g. '-_'
    * separators (string of characters)

    Parameters are passed as alternating parameter name and value, so:

    {'tokenize': "unicode61 remove_diacritics 0 tokenchars '-_'"}

    Content-less tables:

    If you don't need the full-text content in it's original form, you can
    specify a content-less table. Searches and auxiliary functions will work
    as usual, but the only values returned when SELECT-ing can be rowid. Also
    content-less tables do not support UPDATE or DELETE.

    External content tables:

    You can set up triggers to sync these, e.g.

    -- Create a table. And an external content fts5 table to index it.
    CREATE TABLE tbl(a INTEGER PRIMARY KEY, b);
    CREATE VIRTUAL TABLE ft USING fts5(b, content='tbl', content_rowid='a');

    -- Triggers to keep the FTS index up to date.
    CREATE TRIGGER tbl_ai AFTER INSERT ON tbl BEGIN
      INSERT INTO ft(rowid, b) VALUES (new.a, new.b);
    END;
    CREATE TRIGGER tbl_ad AFTER DELETE ON tbl BEGIN
      INSERT INTO ft(fts_idx, rowid, b) VALUES('delete', old.a, old.b);
    END;
    CREATE TRIGGER tbl_au AFTER UPDATE ON tbl BEGIN
      INSERT INTO ft(fts_idx, rowid, b) VALUES('delete', old.a, old.b);
      INSERT INTO ft(rowid, b) VALUES (new.a, new.b);
    END;

    Built-in auxiliary functions:

    * bm25(tbl[, weight_0, ... weight_n])
    * highlight(tbl, col_idx, prefix, suffix)
    * snippet(tbl, col_idx, prefix, suffix, ?, max_tokens)
    """
    rowid = RowIDField()

    class Meta:
        extension_module = 'fts5'
    _error_messages = {'field_type': 'Besides the implicit `rowid` column, all columns must be instances of SearchField', 'index': 'Secondary indexes are not supported for FTS5 models', 'pk': 'FTS5 models must use the default `rowid` primary key'}

    @classmethod
    def validate_model(cls):
        if cls._meta.primary_key.name != 'rowid':
            raise ImproperlyConfigured(cls._error_messages['pk'])
        for field in cls._meta.fields.values():
            if not isinstance(field, (SearchField, RowIDField)):
                raise ImproperlyConfigured(cls._error_messages['field_type'])
        if cls._meta.indexes:
            raise ImproperlyConfigured(cls._error_messages['index'])

    @classmethod
    def fts5_installed(cls):
        if sqlite3.sqlite_version_info[:3] < FTS5_MIN_SQLITE_VERSION:
            return False
        tmp_db = sqlite3.connect(':memory:')
        try:
            tmp_db.execute('CREATE VIRTUAL TABLE fts5test USING fts5 (data);')
        except:
            try:
                tmp_db.enable_load_extension(True)
                tmp_db.load_extension('fts5')
            except:
                return False
            else:
                cls._meta.database.load_extension('fts5')
        finally:
            tmp_db.close()
        return True

    @staticmethod
    def validate_query(query):
        """
        Simple helper function to indicate whether a search query is a
        valid FTS5 query. Note: this simply looks at the characters being
        used, and is not guaranteed to catch all problematic queries.
        """
        tokens = _quote_re.findall(query)
        for token in tokens:
            if token.startswith('"') and token.endswith('"'):
                continue
            if set(token) & _invalid_ascii:
                return False
        return True

    @staticmethod
    def clean_query(query, replace=chr(26)):
        """
        Clean a query of invalid tokens.
        """
        accum = []
        any_invalid = False
        tokens = _quote_re.findall(query)
        for token in tokens:
            if token.startswith('"') and token.endswith('"'):
                accum.append(token)
                continue
            token_set = set(token)
            invalid_for_token = token_set & _invalid_ascii
            if invalid_for_token:
                any_invalid = True
                for c in invalid_for_token:
                    token = token.replace(c, replace)
            accum.append(token)
        if any_invalid:
            return ' '.join(accum)
        return query

    @classmethod
    def match(cls, term):
        """
        Generate a `MATCH` expression appropriate for searching this table.
        """
        return match(cls._meta.entity, term)

    @classmethod
    def rank(cls, *args):
        return cls.bm25(*args) if args else SQL('rank')

    @classmethod
    def bm25(cls, *weights):
        return fn.bm25(cls._meta.entity, *weights)

    @classmethod
    def search(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search using selected `term`."""
        return cls.search_bm25(FTS5Model.clean_query(term), weights, with_score, score_alias, explicit_ordering)

    @classmethod
    def search_bm25(cls, term, weights=None, with_score=False, score_alias='score', explicit_ordering=False):
        """Full-text search using selected `term`."""
        if not weights:
            rank = SQL('rank')
        elif isinstance(weights, dict):
            weight_args = []
            for field in cls._meta.sorted_fields:
                if isinstance(field, SearchField) and (not field.unindexed):
                    weight_args.append(weights.get(field, weights.get(field.name, 1.0)))
            rank = fn.bm25(cls._meta.entity, *weight_args)
        else:
            rank = fn.bm25(cls._meta.entity, *weights)
        selection = ()
        order_by = rank
        if with_score:
            selection = (cls, rank.alias(score_alias))
        if with_score and (not explicit_ordering):
            order_by = SQL(score_alias)
        return cls.select(*selection).where(cls.match(FTS5Model.clean_query(term))).order_by(order_by)

    @classmethod
    def _fts_cmd_sql(cls, cmd, **extra_params):
        tbl = cls._meta.entity
        columns = [tbl]
        values = [cmd]
        for key, value in extra_params.items():
            columns.append(Entity(key))
            values.append(value)
        return NodeList((SQL('INSERT INTO'), cls._meta.entity, EnclosedNodeList(columns), SQL('VALUES'), EnclosedNodeList(values)))

    @classmethod
    def _fts_cmd(cls, cmd, **extra_params):
        query = cls._fts_cmd_sql(cmd, **extra_params)
        return cls._meta.database.execute(query)

    @classmethod
    def automerge(cls, level):
        if not 0 <= level <= 16:
            raise ValueError('level must be between 0 and 16')
        return cls._fts_cmd('automerge', rank=level)

    @classmethod
    def merge(cls, npages):
        return cls._fts_cmd('merge', rank=npages)

    @classmethod
    def optimize(cls):
        return cls._fts_cmd('optimize')

    @classmethod
    def rebuild(cls):
        return cls._fts_cmd('rebuild')

    @classmethod
    def set_pgsz(cls, pgsz):
        return cls._fts_cmd('pgsz', rank=pgsz)

    @classmethod
    def set_rank(cls, rank_expression):
        return cls._fts_cmd('rank', rank=rank_expression)

    @classmethod
    def delete_all(cls):
        return cls._fts_cmd('delete-all')

    @classmethod
    def integrity_check(cls, rank=0):
        return cls._fts_cmd('integrity-check', rank=rank)

    @classmethod
    def VocabModel(cls, table_type='row', table=None):
        if table_type not in ('row', 'col', 'instance'):
            raise ValueError('table_type must be either "row", "col" or "instance".')
        attr = '_vocab_model_%s' % table_type
        if not hasattr(cls, attr):

            class Meta:
                database = cls._meta.database
                table_name = table or cls._meta.table_name + '_v'
                extension_module = fn.fts5vocab(cls._meta.entity, SQL(table_type))
            attrs = {'term': VirtualField(TextField), 'doc': IntegerField(), 'cnt': IntegerField(), 'rowid': RowIDField(), 'Meta': Meta}
            if table_type == 'col':
                attrs['col'] = VirtualField(TextField)
            elif table_type == 'instance':
                attrs['offset'] = VirtualField(IntegerField)
            class_name = '%sVocab' % cls.__name__
            setattr(cls, attr, type(class_name, (VirtualModel,), attrs))
        return getattr(cls, attr)