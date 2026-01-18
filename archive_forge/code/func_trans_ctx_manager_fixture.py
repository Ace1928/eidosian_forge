from __future__ import annotations
import sqlalchemy as sa
from .. import assertions
from .. import config
from ..assertions import eq_
from ..util import drop_all_tables_from_metadata
from ... import Column
from ... import func
from ... import Integer
from ... import select
from ... import Table
from ...orm import DeclarativeBase
from ...orm import MappedAsDataclass
from ...orm import registry
@config.fixture(params=[(rollback, second_operation, begin_nested) for rollback in (True, False) for second_operation in ('none', 'execute', 'begin') for begin_nested in (True, False)])
def trans_ctx_manager_fixture(self, request, metadata):
    rollback, second_operation, begin_nested = request.param
    t = Table('test', metadata, Column('data', Integer))
    eng = getattr(self, 'bind', None) or config.db
    t.create(eng)

    def run_test(subject, trans_on_subject, execute_on_subject):
        with subject.begin() as trans:
            if begin_nested:
                if not config.requirements.savepoints.enabled:
                    config.skip_test('savepoints not enabled')
                if execute_on_subject:
                    nested_trans = subject.begin_nested()
                else:
                    nested_trans = trans.begin_nested()
                with nested_trans:
                    if execute_on_subject:
                        subject.execute(t.insert(), {'data': 10})
                    else:
                        trans.execute(t.insert(), {'data': 10})
                    if rollback:
                        nested_trans.rollback()
                    else:
                        nested_trans.commit()
                    if second_operation != 'none':
                        with assertions.expect_raises_message(sa.exc.InvalidRequestError, "Can't operate on closed transaction inside context manager.  Please complete the context manager before emitting further commands."):
                            if second_operation == 'execute':
                                if execute_on_subject:
                                    subject.execute(t.insert(), {'data': 12})
                                else:
                                    trans.execute(t.insert(), {'data': 12})
                            elif second_operation == 'begin':
                                if execute_on_subject:
                                    subject.begin_nested()
                                else:
                                    trans.begin_nested()
                if execute_on_subject:
                    subject.execute(t.insert(), {'data': 14})
                else:
                    trans.execute(t.insert(), {'data': 14})
            else:
                if execute_on_subject:
                    subject.execute(t.insert(), {'data': 10})
                else:
                    trans.execute(t.insert(), {'data': 10})
                if trans_on_subject:
                    if rollback:
                        subject.rollback()
                    else:
                        subject.commit()
                elif rollback:
                    trans.rollback()
                else:
                    trans.commit()
                if second_operation != 'none':
                    with assertions.expect_raises_message(sa.exc.InvalidRequestError, "Can't operate on closed transaction inside context manager.  Please complete the context manager before emitting further commands."):
                        if second_operation == 'execute':
                            if execute_on_subject:
                                subject.execute(t.insert(), {'data': 12})
                            else:
                                trans.execute(t.insert(), {'data': 12})
                        elif second_operation == 'begin':
                            if hasattr(trans, 'begin'):
                                trans.begin()
                            else:
                                subject.begin()
                        elif second_operation == 'begin_nested':
                            if execute_on_subject:
                                subject.begin_nested()
                            else:
                                trans.begin_nested()
        expected_committed = 0
        if begin_nested:
            expected_committed += 1
        if not rollback:
            expected_committed += 1
        if execute_on_subject:
            eq_(subject.scalar(select(func.count()).select_from(t)), expected_committed)
        else:
            with subject.connect() as conn:
                eq_(conn.scalar(select(func.count()).select_from(t)), expected_committed)
    return run_test