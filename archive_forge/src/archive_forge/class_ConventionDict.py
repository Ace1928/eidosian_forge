from __future__ import annotations
import re
from . import events  # noqa
from .base import _NONE_NAME
from .elements import conv as conv
from .schema import CheckConstraint
from .schema import Column
from .schema import Constraint
from .schema import ForeignKeyConstraint
from .schema import Index
from .schema import PrimaryKeyConstraint
from .schema import Table
from .schema import UniqueConstraint
from .. import event
from .. import exc
class ConventionDict:

    def __init__(self, const, table, convention):
        self.const = const
        self._is_fk = isinstance(const, ForeignKeyConstraint)
        self.table = table
        self.convention = convention
        self._const_name = const.name

    def _key_table_name(self):
        return self.table.name

    def _column_X(self, idx, attrname):
        if self._is_fk:
            try:
                fk = self.const.elements[idx]
            except IndexError:
                return ''
            else:
                return getattr(fk.parent, attrname)
        else:
            cols = list(self.const.columns)
            try:
                col = cols[idx]
            except IndexError:
                return ''
            else:
                return getattr(col, attrname)

    def _key_constraint_name(self):
        if self._const_name in (None, _NONE_NAME):
            raise exc.InvalidRequestError('Naming convention including %(constraint_name)s token requires that constraint is explicitly named.')
        if not isinstance(self._const_name, conv):
            self.const.name = None
        return self._const_name

    def _key_column_X_key(self, idx):
        return self._column_X(idx, 'key')

    def _key_column_X_name(self, idx):
        return self._column_X(idx, 'name')

    def _key_column_X_label(self, idx):
        return self._column_X(idx, '_ddl_label')

    def _key_referred_table_name(self):
        fk = self.const.elements[0]
        refs = fk.target_fullname.split('.')
        if len(refs) == 3:
            refschema, reftable, refcol = refs
        else:
            reftable, refcol = refs
        return reftable

    def _key_referred_column_X_name(self, idx):
        fk = self.const.elements[idx]
        return fk.column.name

    def __getitem__(self, key):
        if key in self.convention:
            return self.convention[key](self.const, self.table)
        elif hasattr(self, '_key_%s' % key):
            return getattr(self, '_key_%s' % key)()
        else:
            col_template = re.match('.*_?column_(\\d+)(_?N)?_.+', key)
            if col_template:
                idx = col_template.group(1)
                multiples = col_template.group(2)
                if multiples:
                    if self._is_fk:
                        elems = self.const.elements
                    else:
                        elems = list(self.const.columns)
                    tokens = []
                    for idx, elem in enumerate(elems):
                        attr = '_key_' + key.replace('0' + multiples, 'X')
                        try:
                            tokens.append(getattr(self, attr)(idx))
                        except AttributeError:
                            raise KeyError(key)
                    sep = '_' if multiples.startswith('_') else ''
                    return sep.join(tokens)
                else:
                    attr = '_key_' + key.replace(idx, 'X')
                    idx = int(idx)
                    if hasattr(self, attr):
                        return getattr(self, attr)(idx)
        raise KeyError(key)